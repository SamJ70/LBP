import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Sliders, History, Cpu, Zap } from 'lucide-react'

const navItems = [
  { to: '/',         icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/optimize', icon: Sliders,         label: 'Optimize' },
  { to: '/history',  icon: History,         label: 'History' },
  { to: '/models',   icon: Cpu,             label: 'AI Models' },
]

export default function Layout({ children }) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="sidebar-logo-row">
            <div className="logo-icon"><Zap size={14} /></div>
            <span className="logo-title">MachineIQ</span>
          </div>
          <p className="logo-sub">ML Machining Optimizer<br />IIT Roorkee · MIC-300</p>
        </div>

        <nav className="sidebar-nav">
          <p className="section-header">Navigation</p>
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to} to={to} end={to === '/'}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              <Icon size={15} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <p>v1.0.0 · Spring 2025-26</p>
          <p>Dept. of ME &amp; IE</p>
        </div>
      </aside>

      <main className="main-content">{children}</main>
    </div>
  )
}