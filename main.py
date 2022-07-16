import os
import re
import sys
import cv2
import onnx
import platform
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

ort.set_default_logger_severity(3)


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH  = os.path.join(BASE_PATH, 'input')
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
MODEL_PATH  = os.path.join(BASE_PATH, 'models')


def breaker(num: int = 50, char: str = "*") -> None:
    print("\n" + num*char + "\n")


def get_image(path: str) -> np.ndarray:
    return cv2.cvtColor(src=cv2.imread(path, cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB)


def show_images(
    image_1: np.ndarray,
    image_2: np.ndarray, 
    cmap_1: str="gnuplot2",
    cmap_2: str="gnuplot2",
    title_1: str="Original",
    title_2: str=None,
    ) -> None:

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_1, cmap=cmap_1)
    plt.axis("off")
    if title_1: plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2, cmap=cmap_2)
    plt.axis("off")
    if title_2: plt.title(title_2)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()


class Model(object):
    def __init__(self, model_type: str) -> None:
        self.ort_session = None
        self.model_type = model_type
        self.size: int = 224
        self.path: str = os.path.join(MODEL_PATH, self.model_type + ".onnx")
    
    def setup(self) -> None:
        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)
    
    def infer(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        input = {self.ort_session.get_inputs()[0].name : image.astype("float32")}
        result = self.ort_session.run(None, input)
        result = np.array(result).squeeze().transpose(1, 2, 0)
        result = np.clip(result * 255, 0, 255).astype("uint8")
        return cv2.resize(src=result, dsize=(w, h), interpolation=cv2.INTER_AREA)


def main():
    args_1: tuple = ("--mode", "-m")
    args_2: tuple = ("--model", "-mo")
    args_3: tuple = ("--filename", "-f")
    args_4: tuple = ("--downscale", "-ds")
    args_5: tuple = ("--negative", "-n")
    args_6: tuple = ("--save", "-s")

    mode: str = "image"
    model_type: str = "candy"
    filename: str = "Test_1.jpg"
    downscale: float = None
    negative: bool = False
    save: bool = False

    if args_1[0] in sys.argv: mode = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: mode = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2[0] in sys.argv: model_type = sys.argv[sys.argv.index(args_2[0]) + 1]
    if args_2[1] in sys.argv: model_type = sys.argv[sys.argv.index(args_2[1]) + 1]

    if args_3[0] in sys.argv: filename = sys.argv[sys.argv.index(args_3[0]) + 1]
    if args_3[1] in sys.argv: filename = sys.argv[sys.argv.index(args_3[1]) + 1]

    if args_4[0] in sys.argv: downscale = float(sys.argv[sys.argv.index(args_4[0]) + 1])
    if args_4[1] in sys.argv: downscale = float(sys.argv[sys.argv.index(args_4[1]) + 1])

    if args_5[0] in sys.argv or args_5[1] in sys.argv: negative = True

    if args_6[0] in sys.argv or args_6[1] in sys.argv: save = True

    assert model_type + ".onnx" in os.listdir(MODEL_PATH), "Model file not found"

    model = Model(model_type=model_type)
    model.setup()

    if re.match(r"image", mode, re.IGNORECASE):
        image = get_image(os.path.join(INPUT_PATH, filename))
        if downscale:
                image = cv2.resize(
                    src=image, 
                    dsize=(int(image.shape[1]/downscale), int(image.shape[0]/downscale)), 
                    interpolation=cv2.INTER_AREA
                )
        result = model.infer(image)
        if negative: result = 255 - result
        if save: cv2.imwrite(os.path.join(OUTPUT_PATH, filename[:-4] + " - Result.png"), cv2.cvtColor(src=result, code=cv2.COLOR_BGR2RGB))
        else: show_images(image_1=image, image_2=result, title_2=f"{model_type.title()} Style Transfer")
    
    elif re.match(r"video", mode, re.IGNORECASE):
        cap = cv2.VideoCapture(os.path.join(INPUT_PATH, filename))

        while True:
            ret, frame = cap.read()
            if ret: 
                if downscale:
                    frame = cv2.resize(
                        src=frame, 
                        dsize=(int(frame.shape[1]/downscale), int(frame.shape[0]/downscale)), 
                        interpolation=cv2.INTER_AREA
                    )
                result = model.infer(frame)
                if negative: result = 255 - result     
                frame = np.concatenate((frame, cv2.cvtColor(src=result, code=cv2.COLOR_BGR2RGB)), axis=1)
                cv2.imshow("Feed", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"): 
                    break
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cap.release()
        cv2.destroyAllWindows()

    elif re.match(r"realtime", mode, re.IGNORECASE):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while True:
            ret, frame = cap.read()
            if not ret: break
            result = model.infer(cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB))       
            if negative: result = 255 - result     
            frame = np.concatenate((frame, cv2.cvtColor(src=result, code=cv2.COLOR_BGR2RGB)), axis=1)
            cv2.imshow(f"{model_type.title()} Style Transfer", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        cap.release()
        cv2.destroyAllWindows()

    else:
        breaker()
        print("--- Unknown Mode ---\n".upper())
        breaker()



if __name__ == "__main__":
    sys.exit(main() or 0)