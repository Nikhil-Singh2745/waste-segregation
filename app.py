"""Demo app. Run `python app.py`, then open http://127.0.0.1:7860 in your browser."""
import cv2
import gradio as gr

from wastesort.config import CLASSES, COLORS, NICE_NAMES
from wastesort.infer import WasteSorter

sorter = WasteSorter()


def run(photo, conf):
    empty = [None, "Upload a photo first.", *[[] for _ in CLASSES]]
    if photo is None:
        return empty
    bgr = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)
    res = sorter.predict(bgr, conf)
    bins = sorter.bins(bgr, res["objects"])
    total = len(res["objects"])
    rows = "".join(f"| {nice} | **{res['counts'][name]}** |\n" for name, nice in zip(CLASSES, NICE_NAMES))
    summary = f"### Found {total} item{'s' * (total != 1)}\n| Bin | Count |\n|---|---|\n{rows}"
    galleries = [[cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in bins[name]] for name in CLASSES]
    return [cv2.cvtColor(res["image"], cv2.COLOR_BGR2RGB), summary, *galleries]


with gr.Blocks(title="Waste Sorter") as demo:
    gr.Markdown("# Waste Sorter\nTake or upload a photo of mixed waste. A CNN finds each item, a second CNN "
                "decides which bin it belongs in. Both networks were trained from scratch.")
    with gr.Row():
        with gr.Column():
            photo = gr.Image(label="Photo", type="numpy", sources=["upload", "webcam", "clipboard"], height=420)
            conf = gr.Slider(0.1, 0.9, value=round(sorter.default_conf, 2), step=0.05, label="Detector confidence",
                             info="Lower = finds more items but more false boxes. Higher = only sure ones.")
            btn = gr.Button("Sort the waste", variant="primary")
        with gr.Column():
            annotated = gr.Image(label="Detected items", height=420)
            summary = gr.Markdown()
    gr.Markdown("## Sorted into bins")
    with gr.Row():
        galleries = [gr.Gallery(label=nice, columns=3, height=260, object_fit="contain") for nice in NICE_NAMES]
    btn.click(run, [photo, conf], [annotated, summary, *galleries])
    photo.upload(run, [photo, conf], [annotated, summary, *galleries])

if __name__ == "__main__":
    demo.launch()
