import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

document.title = 'Logic-stics | Predictive Supply Chain Digital Twin';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
