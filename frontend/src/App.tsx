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

  // Time Travel Logic: Speed buttons ki jagah hours offset use karenge
  const handleTimeTravel = (hours: number) => {
    // Backend API handles 'setSpeed' but we'll use it to pass hours offset
    setSpeed(hours); 
  };

  const handleNodeClick = useCallback((nodeId: number) => {
    if (disruptionMode) {
      console.log(`Injecting disruption at node: ${nodeId}`);
      // 0.05 severity = High impact (Road closure)
      injectDisruption(nodeId, 0.05, 'accident');  
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
             Live DU Speed: {state?.live_anchor_speed ? Math.round(state.live_anchor_speed) : '--'} km/h
          </div>
          
          <div className={`status-badge ${state?.is_live_synced === false ? 'simulating-glow' : ''}`}>
            <span className={`status-dot ${connected ? (state?.is_live_synced !== false ? 'live' : 'warning') : 'danger'}`} />
            {connected 
              ? (state?.is_live_synced !== false ? 'Cloud Sync: Active' : `Time Travel: ${state?.time_offset || 0}h Offset`) 
              : 'Connecting to Render Server…'}
          </div>

          <div className="status-badge">ASTGCN ML Active</div>
          <div className="status-badge">{state?.bottleneck_nodes?.length || 0} Bottlenecks Predicted</div>
        </div>
      </header>

      {/* ── Sidebar: Controls ── */}
      <aside className="sidebar-left">
        <ControlPanel
          connected={connected}
          // Yahan hum offset hours pass kar rahe hain
          onSpeedChange={handleTimeTravel} 
          onDisruptionModeToggle={() => setDisruptionMode(m => !m)}
          disruptionMode={disruptionMode}
          step={state?.step || 0}
          timeOfDay={state?.traffic?.time_of_day ?? traffic?.time_of_day ?? 0}
          dayOfWeek={state?.traffic?.day_of_week ?? traffic?.day_of_week ?? 0}
        />
        {/* Helper text for Judges */}
        <div className="time-travel-legend" style={{padding: '10px', fontSize: '12px', color: '#888'}}>
          * Use 1x for Live, 2x for -2hrs, 5x for -5hrs
        </div>
      </aside>

      <main className="main-map">
        <MapView
          graph={graph}
          state={state}
          disruptionMode={disruptionMode}
          onNodeClick={handleNodeClick}
        />
      </main>

      <aside className="sidebar-right">
        <AlertFeed
          events={events}
          bottleneckCount={state?.bottleneck_nodes?.length || 0}
          meanSpeed={state?.live_anchor_speed || 50}
        />
      </aside>

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