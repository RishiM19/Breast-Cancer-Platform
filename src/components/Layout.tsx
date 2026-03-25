import { LayoutDashboard, Users, BarChart3, Settings, Sun, Moon } from 'lucide-react';
import { ReactNode } from 'react';
import { motion } from 'motion/react';

interface LayoutProps {
  children: ReactNode;
  currentScreen: string;
  onNavigate: (screen: string) => void;
  userName: string;
  isDimmed?: boolean;
  isDarkMode: boolean;
  toggleTheme: () => void;
}

export function Layout({
  children,
  currentScreen,
  onNavigate,
  userName,
  isDimmed = false,
  isDarkMode,
  toggleTheme,
}: LayoutProps) {
  const getInitials = (name: string) => {
    const parts = name.split(' ');
    if (parts.length >= 2) {
      return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    }
    return name.substring(0, 2).toUpperCase();
  };

  const menuItems = [
    { icon: LayoutDashboard, label: 'Dashboard', screen: 'dashboard' },
    { icon: Users, label: 'Patient Database', screen: 'patient-list' },
    { icon: BarChart3, label: 'Model Stats', screen: 'model-stats' },
    { icon: Settings, label: 'Settings', screen: 'settings' },
  ];

  return (
    <div className="min-h-screen bg-[#F8F9FB] dark:bg-gray-900 flex">
      <motion.div
        className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col shrink-0"
        initial={{ opacity: 0, x: -16 }}
        animate={{ opacity: isDimmed ? 0.5 : 1, x: 0 }}
        transition={{ duration: 0.35, ease: 'easeOut' }}
      >
        {/* Logo */}
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-[#007AFF] rounded-lg flex items-center justify-center">
              <BarChart3 className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-gray-900 dark:text-white">Lumina</h2>
              <p className="text-xs text-gray-500 dark:text-gray-400">Breast AI</p>
            </div>
          </div>
        </div>

        {/* Navigation Menu */}
        <nav className="flex-1 p-4">
          <div className="space-y-2">
            {menuItems.map((item, index) => {
              const isActive = currentScreen === item.screen;
              return (
                <div key={index}>
                  <button
                    onClick={() => onNavigate(item.screen)}
                    className={`w-full flex items-center justify-between px-4 py-3 rounded-lg transition-all ${
                      isActive
                        ? 'bg-[#007AFF] text-white shadow-md'
                        : 'text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <item.icon className="w-5 h-5" />
                      <span>{item.label}</span>
                    </div>
                  </button>
                </div>
              );
            })}
          </div>
        </nav>

        {/* Theme toggle */}
        <div className="px-4 pb-2">
          <button
            type="button"
            onClick={toggleTheme}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50"
            aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {isDarkMode ? (
              <>
                <Sun className="w-5 h-5 shrink-0 text-amber-400" />
                <span>Light mode</span>
              </>
            ) : (
              <>
                <Moon className="w-5 h-5 shrink-0" />
                <span>Dark mode</span>
              </>
            )}
          </button>
        </div>

        {/* User Profile */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-gray-50 dark:bg-gray-700/40">
            <div className="w-10 h-10 bg-gradient-to-br from-[#007AFF] to-[#0062CC] rounded-full flex items-center justify-center text-white">
              {getInitials(userName)}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-gray-900 dark:text-white text-sm truncate">{userName}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Senior Radiologist</p>
            </div>
          </div>
        </div>
      </motion.div>

      <div className="flex-1 flex flex-col min-h-0 overflow-auto">{children}</div>
    </div>
  );
}
