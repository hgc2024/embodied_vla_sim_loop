import type { FrameMessage } from "../types";

interface Props {
  frame: FrameMessage | null;
}

export function CameraPanel({ frame }: Props) {
  return (
    <section className="panel camera-panel">
      <h2>Wrist camera</h2>
      <p className="muted small camera-hint">
        Mounted on the arm itself, not watching it from across the room -- so it's
        often an extreme close-up of whatever surface is nearest. That's normal.
      </p>
      {frame === null ? (
        <p className="muted">Waiting for the first observation…</p>
      ) : (
        <>
          <p className="task-instruction">"{frame.task_instruction}"</p>
          <div className="camera-grid">
            <figure>
              <img src={`data:image/jpeg;base64,${frame.rgb_jpeg}`} alt="RGB camera feed" />
              <figcaption>RGB (color)</figcaption>
            </figure>
            <figure>
              <img src={`data:image/jpeg;base64,${frame.depth_jpeg}`} alt="Depth camera feed" />
              <figcaption
                className="hint"
                title="Same view, colored by distance instead of color: purple is near, red is far (up to 3m)."
              >
                Depth (colored by distance)
              </figcaption>
            </figure>
          </div>
          <p className="muted small">frame #{frame.frame_id}</p>
        </>
      )}
    </section>
  );
}
