import { useState } from "react";

const DISMISSED_KEY = "embodied-vla-info-banner-collapsed";

export function InfoBanner() {
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(DISMISSED_KEY) === "1",
  );

  function toggle() {
    const next = !collapsed;
    setCollapsed(next);
    localStorage.setItem(DISMISSED_KEY, next ? "1" : "0");
  }

  return (
    <section className="panel info-banner">
      <button className="info-banner-toggle" onClick={toggle}>
        <span>What am I looking at?</span>
        <span className="chevron">{collapsed ? "▸" : "▾"}</span>
      </button>
      {!collapsed && (
        <div className="info-banner-body">
          <p>
            This is a simulated robot arm, plus a second program deciding how to
            move it (the "policy"), shown live. The two run independently so the
            physics never has to pause and wait for the policy to think.
          </p>
          <p>
            The arm here is reaching toward the orange block using a
            <strong> hand-written "reach toward it" controller</strong>, not a
            trained AI yet -- that's a deliberate placeholder so the dashboard
            shows real, goal-directed motion before any model exists. The
            robot itself is also a simple placeholder shape, not a real robot
            model. See the README's "In plain terms" section for more,
            including what each panel below means.
          </p>
        </div>
      )}
    </section>
  );
}
