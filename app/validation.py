import io
from PIL import Image

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
MIN_DIMENSION = 10  # Minimum 10x10 pixels


class ValidationError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def validate_image_payload(file_storage) -> bytes:
    """
    Validates uploaded multipart file against security and format rules:
    - Verifies non-empty file payload
    - Enforces extension allowlist (.jpg, .jpeg, .png, .webp)
    - Enforces 10MB payload size boundary
    - Verifies magic bytes/structural header integrity via Pillow
    - Rejects micro-images (<10x10 pixels)
    """
    if file_storage is None or file_storage.filename == "":
        raise ValidationError("No file uploaded or filename is empty.", status_code=400)

    filename = file_storage.filename
    if "." not in filename:
        raise ValidationError("File lacks an extension.", status_code=400)

    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            status_code=415
        )

    image_bytes = file_storage.read()

    if len(image_bytes) == 0:
        raise ValidationError("Uploaded file is empty (0 bytes).", status_code=400)

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValidationError("File size exceeds 10 MB limit.", status_code=413)

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except Exception as e:
        raise ValidationError(f"Corrupted or invalid image file: {str(e)}", status_code=400)

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            if width < MIN_DIMENSION or height < MIN_DIMENSION:
                raise ValidationError(
                    f"Image dimensions ({width}x{height}) are too small. Minimum required: {MIN_DIMENSION}x{MIN_DIMENSION}.",
                    status_code=400
                )
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Unable to inspect image geometry: {str(e)}", status_code=400)

    return image_bytes