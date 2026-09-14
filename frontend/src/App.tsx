import "./App.css";
import { CameraPanel } from "./components/CameraPanel";
import { ConnectionBadge } from "./components/ConnectionBadge";
import { ControlsPanel } from "./components/ControlsPanel";
import { InfoBanner } from "./components/InfoBanner";
import { JointStatePanel } from "./components/JointStatePanel";
import { TelemetryPanel } from "./components/TelemetryPanel";
import { useDashboardSocket } from "./useDashboardSocket";

function App() {
  const { connected, frame, overview, status } = useDashboardSocket();

  return (
    <div className="app">
      <header className="app-header">
        <h1>embodied-vla-sim-loop dashboard</h1>
        <ConnectionBadge connected={connected} />
      </header>

      <InfoBanner />

      <main className="app-grid">
        <CameraPanel frame={frame} overview={overview} />
        <div className="sidebar">
          <TelemetryPanel status={status} />
          <ControlsPanel status={status} />
          <JointStatePanel frame={frame} />
        </div>
      </main>
    </div>
  );
}

export default App;
