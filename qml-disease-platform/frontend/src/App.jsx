import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar.jsx'
import Home from './pages/Home.jsx'
import DiseasePage from './pages/DiseasePage.jsx'
import BrainTumorPage from './pages/BrainTumorPage.jsx'
import PerformancePage from './pages/PerformancePage.jsx'
import GeneralPage from './pages/GeneralPage.jsx'

export default function App() {
  return (
    <div className="app-shell">
      <Navbar />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/general" element={<GeneralPage />} />
          <Route path="/disease/brain_tumor" element={<BrainTumorPage />} />
          <Route path="/disease/:diseaseId" element={<DiseasePage />} />
          <Route path="/disease/:diseaseId/performance" element={<PerformancePage />} />
        </Routes>
      </main>
      <footer className="app-footer">
        Qure AI · Hybrid Quantum ML Disease Detection Platform · SIH26139 · Prototype for research &amp; educational use only
      </footer>
    </div>
  )
}
