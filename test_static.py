# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import io

# Consolidate drawing utilities in `face_utils.py` to avoid duplication.
from face_utils import draw_landmarks_on_image

# STEP 2: Create an FaceLandmarker object.
def run_face_landmarker(image_path: str = "PP_Rudy.jpeg",
                        model_asset_path: str = "face_landmarker_v2_with_blendshapes.task",
                        num_faces: int = 1,
                        output_path: str | None = None):
    """Run the FaceLandmarker on `image_path` using the given model file.

    Returns the annotated image as a NumPy array (RGB). If running inside a
    Jupyter notebook, the image will be displayed inline. If `output_path` is
    provided or inline display is unavailable, the annotated image will be
    written to that path (or to `annotated_<basename>.png`).
    """
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=num_faces)
    detector = vision.FaceLandmarker.create_from_options(options)

    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)

    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    # Try to display inline in Jupyter using IPython.display. Fallback to file if unavailable.
    try:
        from IPython.display import display, Image as IPyImage
        bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        retval, buf = cv2.imencode('.png', bgr)
        if retval:
            display(IPyImage(data=buf.tobytes()))
        else:
            raise RuntimeError('cv2.imencode failed')
    except Exception:
        if output_path is None:
            out_path = f"output/annotated_{image_path.rsplit('/',1)[-1].rsplit('.',1)[0]}.jpg"
        else:
            out_path = output_path
        cv2.imwrite(out_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"Annotated image written to {out_path}")

    return annotated_image


if __name__ == '__main__':
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else 'dataset/pasfoto.jpg'
    model = sys.argv[2] if len(sys.argv) > 2 else 'face_landmarker_v2_with_blendshapes.task'
    run_face_landmarker(img, model)