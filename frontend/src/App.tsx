import { useState, useCallback } from 'react';
import { useSimulation } from './hooks/useSimulation';
import MapView from './components/MapView';
import ControlPanel from './components/ControlPanel';
import AlertFeed from './components/AlertFeed';
import KPIDashboard from './components/KPIDashboard';
import './index.css';

export default function App() {
  const { state, graph, connected, events, injectDisruption, setSpeed } = useSimulation();
  const [disruptionMode, setDisruptionMode] = useState(false);

  const handleNodeClick = useCallback((nodeId: number) => {
    if (disruptionMode) {
      injectDisruption(nodeId, 0.05, 'accident');  // 0.05 = 5% of normal speed (gridlock)
      setDisruptionMode(false);
    }
  }, [disruptionMode, injectDisruption]);

  const fleet = state?.fleet;
  const traffic = state?.traffic;

  return (
    <div className="app-layout">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="logo">
          <div className="logo-icon">LG</div>
          <div>
            <div className="logo-text">Logic-stics</div>
            <div className="logo-subtitle">Predictive Digital Twin</div>
          </div>
        </div>
        <div className="header-status">
          <div className="status-badge">
            <span className={`status-dot ${connected ? 'live' : 'danger'}`} />
            {connected ? 'Live' : 'Reconnecting…'}
          </div>
          <div className="status-badge">
            🧠 ASTGCN Active
          </div>
          <div className="status-badge">
            📡 {state?.bottleneck_nodes?.length || 0} Bottlenecks Detected
          </div>
        </div>
      </header>

      {/* ── Left Sidebar: Controls ── */}
      <aside className="sidebar-left">
        <ControlPanel
          connected={connected}
          onSpeedChange={setSpeed}
          onDisruptionModeToggle={() => setDisruptionMode(m => !m)}
          disruptionMode={disruptionMode}
          step={state?.step || 0}
          timeOfDay={traffic?.time_of_day || 0}
          dayOfWeek={traffic?.day_of_week || 0}
        />
      </aside>

      {/* ── Main Map ── */}
      <main className="main-map">
        <MapView
          graph={graph}
          state={state}
          disruptionMode={disruptionMode}
          onNodeClick={handleNodeClick}
        />
      </main>

      {/* ── Right Sidebar: Alerts ── */}
      <aside className="sidebar-right">
        <AlertFeed
          events={events}
          bottleneckCount={state?.bottleneck_nodes?.length || 0}
          meanSpeed={state?.prediction?.mean_predicted_speed || (traffic?.speeds ? traffic.speeds.reduce((a, b) => a + b, 0) / traffic.speeds.length : 50)}
        />
      </aside>

      {/* ── Bottom: KPIs ── */}
      <div className="bottom-panel">
        <KPIDashboard
          activeShipments={fleet?.active_count || 0}
          onTimeRate={fleet && fleet.total_deliveries > 0 ? Math.min(99, (fleet.total_deliveries / (fleet.total_deliveries + fleet.total_reroutes + 1)) * 100) : 95}
          totalReroutes={fleet?.total_reroutes || 0}
          totalDeliveries={fleet?.total_deliveries || 0}
          timeSaved={fleet?.total_time_saved || 0}
          avgDeliveryTime={fleet?.avg_delivery_time || 0}
        />
      </div>
    </div>
  );
}
