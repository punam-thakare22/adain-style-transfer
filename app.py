import os
import shutil
import time
import torch
# A free hosting tier's CPU is usually a small fractional slice (not a full core), and by
# default PyTorch spins up one thread per CPU it *thinks* it has, which thrashes and uses
# more memory/time than it saves on a host like that. Pinning to 1 thread, before anything
# else touches torch, keeps this fast and light on a constrained host. Harmless locally too.
torch.set_num_threads(1)
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import io

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


# Everything is resolved relative to this file, so the app runs from any working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24).hex()
# ^ Uses a real SECRET_KEY from the environment when one is set (do this for any public
# deployment — it signs session cookies and CSRF tokens). Falls back to a random key
# generated fresh each process start, which is fine for local/portfolio use but means
# sessions won't survive a server restart. Never commit a hardcoded key to a public repo.
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024  # 15 MB per request; enforced by Flask/Werkzeug before the view runs
MAX_UPLOAD_MB = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

RECENT_CREATIONS_LIMIT = 8
RESOLUTION_CHOICES = ('256', '512', '1024')  # shorter-edge size in px; the network is fully convolutional so any of these runs for real

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    resolution = HiddenField(default='512')
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VGGEncoder(os.path.join(BASE_DIR, 'vgg_normalised.pth')).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(
    os.path.join(BASE_DIR, 'experiment', 'final_exp', 'decoder_final.pth'),
    map_location=device,
))

encoder.eval()
decoder.eval()


def _count_params(module):
    """Real parameter count read from the loaded model, or None if it can't be introspected."""
    try:
        return sum(p.numel() for p in module.parameters())
    except Exception:
        return None


MODEL_INFO = {
    'architecture': 'AdaIN style transfer (Huang & Belongie, ICCV 2017)',
    'encoder': 'VGG-19, first 4 blocks (relu1_1–relu4_1), frozen, ImageNet-pretrained',
    'decoder_params': _count_params(decoder),
    'device': str(device).upper(),
    # If encoder/decoder had failed to load, the import above would have raised and this
    # module-level code would never run — so reaching this line means the model really is ready.
    'status': 'Model ready',
}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def validate_uploaded_image(path):
    """Confirm a saved upload is really a decodable PNG/JPEG, not just a file with the right extension."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def list_recent_creations(limit=RECENT_CREATIONS_LIMIT):
    """Real creations read from disk: every 'stylized_*' file in the uploads folder, newest first."""
    folder = app.config['UPLOAD_FOLDER']
    if not os.path.isdir(folder):
        return []
    items = []
    for name in os.listdir(folder):
        if not name.startswith('stylized_') or not allowed_file(name):
            continue
        path = os.path.join(folder, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        items.append({'filename': name, 'mtime': mtime})
    items.sort(key=lambda it: it['mtime'], reverse=True)
    for it in items:
        it['created'] = datetime.fromtimestamp(it['mtime']).strftime('%b %d, %Y %H:%M')
    return items[:limit]


# ---------------------------------------------------------------------------
# Sample ("preset") images shown as one-click thumbnails in the UI.
# Only images that have a thumbnail in static/presets/<kind>/ are offered, and
# the browser can only ever refer to them by id, never by path.
# ---------------------------------------------------------------------------
PRESET_SOURCES = {
    'content': os.path.join(BASE_DIR, 'content_data'),
    'style': os.path.join(BASE_DIR, 'style_data'),
}
PRESET_THUMBS = os.path.join(BASE_DIR, 'static', 'presets')
PRESET_LABELS = {'the_resevoir_at_poitiers': 'The Reservoir at Poitiers'}

# Curatorial tags for the style gallery filter chips. This is hand-authored metadata about
# the actual paintings in style_data/ (not a model prediction), so it's safe to show as fact.
STYLE_CATEGORIES = {
    'picasso_seated_nude_hr': 'Portrait',
    'picasso_self_portrait': 'Portrait',
    'la_muse': 'Portrait',
    'woman_with_hat_matisse': 'Portrait',
    'woman_in_peasant_dress': 'Portrait',
    'mondrian': 'Abstract',
    'contrast_of_forms': 'Abstract',
    'goeritz': 'Abstract',
    'antimonocromatismo': 'Abstract',
    'impronte_d_artista': 'Abstract',
    'sketch': 'Sketch',
    'scene_de_rue': 'Sketch',
    'brushstrokes': 'Painterly',
    'en_campo_gris': 'Painterly',
    'flower_of_life': 'Painterly',
    'the_resevoir_at_poitiers': 'Painterly',
    'trial': 'Painterly',
}


def load_presets(kind):
    thumb_dir = os.path.join(PRESET_THUMBS, kind)
    if not os.path.isdir(thumb_dir) or not os.path.isdir(PRESET_SOURCES[kind]):
        return []
    sources = {os.path.splitext(f)[0]: f for f in os.listdir(PRESET_SOURCES[kind]) if allowed_file(f)}
    presets = []
    for thumb in sorted(os.listdir(thumb_dir)):
        stem = os.path.splitext(thumb)[0]
        if stem in sources:
            presets.append({
                'id': stem,
                'file': sources[stem],
                'label': PRESET_LABELS.get(stem, stem.replace('_', ' ').title()),
                'category': STYLE_CATEGORIES.get(stem, 'Painterly') if kind == 'style' else None,
            })
    return presets


def diagnose_presets(kind):
    """Real, on-disk diagnosis of why a preset list might be empty — shown on the page itself
    instead of a silent '0 styles' so this is never a guessing game. Not shown when presets load fine."""
    thumb_dir = os.path.join(PRESET_THUMBS, kind)
    source_dir = PRESET_SOURCES[kind]
    return {
        'thumb_dir': thumb_dir,
        'thumb_dir_exists': os.path.isdir(thumb_dir),
        'thumb_count': len(os.listdir(thumb_dir)) if os.path.isdir(thumb_dir) else 0,
        'source_dir': source_dir,
        'source_dir_exists': os.path.isdir(source_dir),
        'source_count': len(os.listdir(source_dir)) if os.path.isdir(source_dir) else 0,
    }


PRESETS = {kind: load_presets(kind) for kind in PRESET_SOURCES}
# Computed once at startup, same as PRESETS: if you fix the folders on disk, you must restart
# the server process for this (and PRESETS) to pick the change up — editing files alone won't.
PRESET_DEBUG = {kind: diagnose_presets(kind) for kind in PRESET_SOURCES if not PRESETS[kind]}


def resolve_preset(kind, value):
    """Turn a 'preset:<id>' form value into a file inside the uploads folder.

    Returns the filename, or None if the value is not a known preset."""
    if not value or not value.startswith('preset:'):
        return None
    preset_id = value[len('preset:'):]
    match = next((p for p in PRESETS[kind] if p['id'] == preset_id), None)
    if match is None:
        return None
    filename = 'preset_%s_%s%s' % (kind, match['id'], os.path.splitext(match['file'])[1].lower())
    destination = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(destination):
        shutil.copyfile(os.path.join(PRESET_SOURCES[kind], match['file']), destination)
    return filename


def previous_upload(value):
    """Sanitise a filename posted back in the hidden fields (must already exist in uploads)."""
    name = secure_filename(value or '')
    if name and os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], name)):
        return name
    return None


def style_transfer(content_image, style_image, encoder, decoder, alpha, device, resolution=512):
    content_transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)



def _save_and_validate_upload(file_storage, kind):
    """Save an uploaded file and confirm it's a real image. Returns (filename, error)."""
    if not allowed_file(file_storage.filename):
        return None, '%s file must be a PNG or JPG image.' % kind.capitalize()
    filename = secure_filename(file_storage.filename)
    dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_storage.save(dest)
    if not validate_uploaded_image(dest):
        os.remove(dest)
        return None, 'That %s file is not a valid PNG or JPG image.' % kind
    return filename, None


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None
    insights = None
    upload_error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            content_filename, upload_error = _save_and_validate_upload(form.content.data, 'content')
            form.content_path.data = content_filename or ''
        else:
            content_filename = resolve_preset('content', form.content_path.data) or previous_upload(form.content_path.data)
            form.content_path.data = content_filename or ''

        if form.style.data and form.style.data.filename:
            style_filename, style_upload_error = _save_and_validate_upload(form.style.data, 'style')
            upload_error = upload_error or style_upload_error
            form.style_path.data = style_filename or ''
        else:
            style_filename = resolve_preset('style', form.style_path.data) or previous_upload(form.style_path.data)
            form.style_path.data = style_filename or ''

        if upload_error:
            error = upload_error
        elif content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')
                content_dims = content_image.size  # (W, H), real dimensions of the uploaded/selected file
                style_dims = style_image.size

                alpha = float(form.alpha.data)
                alpha = max(0.0, min(1.0, alpha))  # defensive clamp; the UI slider already restricts to [0, 1]

                resolution_choice = form.resolution.data if form.resolution.data in RESOLUTION_CHOICES else '512'
                resolution = int(resolution_choice)

                start = time.perf_counter()
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device,
                                                 resolution=resolution)
                elapsed = time.perf_counter() - start

                result_filename = 'stylized_' + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)

                result_image = result_filename
                # stylized_image is [1, 3, H, W]; AdaIN's encoder/decoder don't change spatial size,
                # so this is exactly the output resolution that was saved.
                output_h, output_w = stylized_image.shape[-2], stylized_image.shape[-1]
                insights = {
                    'content_dims': '%d × %d px' % content_dims,
                    'style_dims': '%d × %d px' % style_dims,
                    'output_dims': '%d × %d px' % (output_w, output_h),
                    'elapsed_s': round(elapsed, 2),
                    'elapsed_ms': round(elapsed * 1000),
                    'alpha': alpha,
                    'resolution': resolution_choice,
                    'device': MODEL_INFO['device'],
                }
            except Exception as e:
                error = str(e)
        elif not content_filename and not style_filename:
            error = 'Please choose a content image and a style image (PNG or JPG), or pick a sample.'
        elif not content_filename:
            error = 'Please choose a content image (PNG or JPG), or pick a sample.'
        else:
            error = 'Please choose a style image (PNG or JPG), or pick a sample.'
    elif request.method == 'POST':  # e.g. expired CSRF token; stay quiet on the first page load (GET)
        error = 'The form could not be submitted. Please reload the page and try again.'

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error, insights=insights, model_info=MODEL_INFO,
                           content_presets=PRESETS['content'], style_presets=PRESETS['style'],
                           recent_creations=list_recent_creations(), max_upload_mb=MAX_UPLOAD_MB,
                           resolution_choices=RESOLUTION_CHOICES, preset_debug=PRESET_DEBUG)


@app.errorhandler(413)
def file_too_large(_e):
    # formdata=None stops Flask-WTF from trying to read request.form/request.files to
    # auto-bind the form: that read is exactly what raised 413 in the first place, and
    # retrying it here would just raise the same error again instead of rendering a page.
    form = UploadForm(formdata=None)
    error = 'That upload is too large. Please use images under %d MB.' % MAX_UPLOAD_MB
    return render_template('index.html', form=form, result_image=None, content_image=None,
                           style_image=None, error=error, insights=None, model_info=MODEL_INFO,
                           content_presets=PRESETS['content'], style_presets=PRESETS['style'],
                           recent_creations=list_recent_creations(), max_upload_mb=MAX_UPLOAD_MB,
                           resolution_choices=RESOLUTION_CHOICES, preset_debug=PRESET_DEBUG), 413


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/download/<filename>')
def download_image(filename):
    """Force a 'Save As' download for a result in uploads/, instead of an inline view."""
    name = secure_filename(filename)
    if not name or not os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], name)):
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], name, as_attachment=True, download_name=name)


@app.route('/samples/<kind>/<preset_id>')
def send_sample(kind, preset_id):
    match = next((p for p in PRESETS.get(kind, []) if p['id'] == preset_id), None)
    if match is None:
        abort(404)
    return send_from_directory(PRESET_SOURCES[kind], match['file'])


@app.route('/examples/<path:filename>')
def send_example(filename):
    # Anchored to BASE_DIR (not a bare relative path) so this still finds the folder
    # no matter what directory the app was launched from.
    return send_from_directory(os.path.join(BASE_DIR, 'examples'), filename)


if __name__ == '__main__':
    # Debug mode (the interactive Werkzeug debugger) lets anyone who can reach the server
    # run arbitrary Python through the browser when an error page is shown. Keep it off
    # unless FLASK_DEBUG=1 is explicitly set for local development, and never set it on
    # a publicly hosted deployment. PORT/HOST are read from the environment so this also
    # works unchanged on hosts like Render/Railway that assign their own port.
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    host = os.environ.get('HOST', '127.0.0.1' if debug_mode else '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    app.run(host=host, port=port, debug=debug_mode)






