import { useState, useCallback } from 'react';
import { useSimulation } from './hooks/useSimulation';
import MapView from './components/MapView';
import ControlPanel from './components/ControlPanel';
import AlertFeed from './components/AlertFeed';
import KPIDashboard from './components/KPIDashboard';
import './index.css';

export default function App() {
  // useSimulation hook returns state, graph, etc.
  const { state, graph, connected, events, injectDisruption, setSpeed } = useSimulation();
  const [disruptionMode, setDisruptionMode] = useState(false);

  // Time Travel Logic: Passing offset hours to the backend via setSpeed
  const handleTimeTravel = (hours: number) => {
    if (setSpeed) {
      setSpeed(hours);
    }
  };

  const handleNodeClick = useCallback((nodeId: number) => {
    if (disruptionMode) {
      console.log(`Injecting disruption at node: ${nodeId}`);
      // 0.05 severity = Significant traffic slowdown
      injectDisruption(nodeId, 0.05, 'accident');  
      setDisruptionMode(false);
    }
  }, [disruptionMode, injectDisruption]);

  // Safe access to nested state properties
  const fleet = state?.fleet;
  const traffic = state?.traffic;
  // Casting state as 'any' to prevent TypeScript build errors on new properties
  const anyState = state as any;

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
             Live DU Speed: {anyState?.live_anchor_speed ? Math.round(anyState.live_anchor_speed) : '--'} km/h
          </div>
          
          <div className={`status-badge ${anyState?.is_live_synced === false ? 'simulating-glow' : ''}`}>
            <span className={`status-dot ${connected ? (anyState?.is_live_synced !== false ? 'live' : 'warning') : 'danger'}`} />
            {connected 
              ? (anyState?.is_live_synced !== false ? 'Cloud Sync: Active' : `Time Travel: ${anyState?.time_offset || 0}h Offset`) 
              : 'Connecting to Render Server…'}
          </div>

          <div className="status-badge">ASTGCN ML Active</div>
          <div className="status-badge">{anyState?.bottleneck_nodes?.length || 0} Bottlenecks Predicted</div>
        </div>
      </header>

      {/* ── Left Sidebar: Controls ── */}
      <aside className="sidebar-left">
        <ControlPanel
          connected={connected}
          onSpeedChange={handleTimeTravel} 
          onDisruptionModeToggle={() => setDisruptionMode(m => !m)}
          disruptionMode={disruptionMode}
          step={anyState?.step || 0}
          timeOfDay={traffic?.time_of_day ?? 0}
          dayOfWeek={traffic?.day_of_week ?? 0}
        />
        <div style={{padding: '12px', fontSize: '11px', color: '#64748b', borderTop: '1px solid #1e293b', marginTop: '10px'}}>
           <strong>Time Sync:</strong> 1x = Live (IST) <br/>
           <strong>Travel:</strong> Use 5x for -5hrs ago
        </div>
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
          bottleneckCount={anyState?.bottleneck_nodes?.length || 0}
          meanSpeed={anyState?.live_anchor_speed || 50}
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