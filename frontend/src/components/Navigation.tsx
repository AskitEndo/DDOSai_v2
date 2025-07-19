import React, { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  Shield,
  Activity,
  Zap,
  FileText,
  Settings,
  Bell,
  Menu,
  X,
  ChevronDown,
  User,
  LogOut,
  HelpCircle,
  Database,
} from "lucide-react";
import { useAppContext } from "../context/AppContext";
import {
  generateRandomDetections,
  generateRandomMetrics,
  generateRandomNetworkGraph,
} from "../utils/dummyData";

// Component to load dummy data (subtle version for online mode)
const LoadDummyDataButton = () => {
  const { dispatch } = useAppContext();
  const [isLoading, setIsLoading] = useState(false);

  const loadDummyData = () => {
    setIsLoading(true);

    // Generate random data
    const detections = generateRandomDetections(50);
    const metrics = generateRandomMetrics();
    const networkGraph = generateRandomNetworkGraph();

    // Update the app state with dummy data
    dispatch({ type: "SET_DETECTIONS", payload: detections });
    dispatch({ type: "SET_METRICS", payload: metrics });
    dispatch({ type: "SET_NETWORK_GRAPH", payload: networkGraph });

    // Short delay just for the button loading state
    setTimeout(() => {
      setIsLoading(false);
    }, 300);
  };

  return (
    <button
      onClick={loadDummyData}
      disabled={isLoading}
      className="px-3 py-1 bg-blue-900/30 hover:bg-blue-800/50 text-blue-300/80 rounded-md flex items-center text-xs transition-colors"
    >
      <Database className="w-3 h-3 mr-1" />
      {isLoading ? "Loading..." : "Load Sample Data"}
    </button>
  );
};

const Navigation: React.FC = () => {
  const location = useLocation();
  const { state } = useAppContext();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);

  const navItems = [
    { path: "/", label: "Dashboard", icon: Activity },
    { path: "/simulation", label: "Simulation", icon: Zap },
    { path: "/logs", label: "Logs", icon: FileText },
  ];

  return (
    <nav className="bg-gradient-to-r from-gray-900 to-gray-800 border-b border-blue-900/30 shadow-lg sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <div className="flex items-center">
            <div className="flex items-center space-x-2">
              <img
                src="/DDOSaiNOBG.png"
                alt="DDoS.AI Logo"
                className="w-10 h-10 object-contain"
              />
              <div className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-300">
                DDoS.AI
              </div>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex ml-8 space-x-4">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;

                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                      isActive
                        ? "bg-blue-500/20 text-blue-400 border border-blue-500/30 shadow-[0_0_10px_rgba(59,130,246,0.3)]"
                        : "text-gray-300 hover:bg-gray-700/50 hover:text-blue-300 hover:border hover:border-blue-800/50"
                    }`}
                  >
                    <Icon className="w-4 h-4 mr-2" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>

          {/* Status and Controls */}
          <div className="hidden md:flex items-center space-x-6">
            {/* Connection Status */}
            <div className="flex items-center space-x-2 text-sm bg-gray-800/50 px-3 py-1.5 rounded-full border border-gray-700">
              <div
                className={`w-2 h-2 rounded-full ${
                  state.isConnected
                    ? "bg-green-500 animate-pulse shadow-[0_0_5px_rgba(34,197,94,0.5)]"
                    : "bg-red-500 shadow-[0_0_5px_rgba(239,68,68,0.5)]"
                }`}
              />
              <span className="text-gray-300">
                {state.isConnected ? "Connected" : "Disconnected"}
              </span>
            </div>

            {/* Threat Level */}
            {state.metrics && (
              <div className="flex items-center space-x-2 text-sm bg-gray-800/50 px-3 py-1.5 rounded-full border border-gray-700">
                <span className="text-gray-300">Threat Level:</span>
                <span
                  className={`font-medium px-2 py-0.5 rounded-md ${
                    state.metrics.threat_level >= 4
                      ? "bg-red-900/50 text-red-300"
                      : state.metrics.threat_level >= 2
                      ? "bg-yellow-900/50 text-yellow-300"
                      : "bg-green-900/50 text-green-300"
                  }`}
                >
                  {state.metrics.threat_level}/5
                </span>
              </div>
            )}

            {/* Load Sample Data Button (subtle version for online mode) */}
            {state.isConnected && (
              <div className="mr-2">
                <LoadDummyDataButton />
              </div>
            )}

            {/* Notifications */}
            <button className="relative p-2 text-gray-300 hover:text-blue-300 hover:bg-gray-700/50 rounded-full transition-colors">
              <Bell className="w-5 h-5" />
              <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full"></span>
            </button>

            {/* User Menu */}
            <div className="relative">
              <button
                onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
                className="flex items-center space-x-2 bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1.5 rounded-md transition-colors"
              >
                <div className="w-6 h-6 rounded-full bg-blue-600 flex items-center justify-center text-xs font-bold">
                  A
                </div>
                <span>Admin</span>
                <ChevronDown className="w-4 h-4" />
              </button>

              {isUserMenuOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-gray-800 rounded-md shadow-lg py-1 border border-gray-700 z-50">
                  <a
                    href="#"
                    className="flex items-center px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white"
                  >
                    <User className="w-4 h-4 mr-2" />
                    Profile
                  </a>
                  <a
                    href="#"
                    className="flex items-center px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white"
                  >
                    <Settings className="w-4 h-4 mr-2" />
                    Settings
                  </a>
                  <a
                    href="#"
                    className="flex items-center px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white"
                  >
                    <HelpCircle className="w-4 h-4 mr-2" />
                    Help
                  </a>
                  <div className="border-t border-gray-700 my-1"></div>
                  <a
                    href="#"
                    className="flex items-center px-4 py-2 text-sm text-red-400 hover:bg-gray-700"
                  >
                    <LogOut className="w-4 h-4 mr-2" />
                    Logout
                  </a>
                </div>
              )}
            </div>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="p-2 rounded-md text-gray-400 hover:text-white hover:bg-gray-700 focus:outline-none"
            >
              {isMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {isMenuOpen && (
        <div className="md:hidden bg-gray-800 shadow-lg border-t border-gray-700">
          <div className="px-2 pt-2 pb-3 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;

              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center px-3 py-2 rounded-md text-base font-medium ${
                    isActive
                      ? "bg-blue-500/20 text-blue-400"
                      : "text-gray-300 hover:bg-gray-700 hover:text-white"
                  }`}
                  onClick={() => setIsMenuOpen(false)}
                >
                  <Icon className="w-5 h-5 mr-3" />
                  {item.label}
                </Link>
              );
            })}

            <div className="pt-4 pb-2 border-t border-gray-700">
              <div className="flex items-center px-3 py-2">
                <div
                  className={`w-2 h-2 rounded-full mr-2 ${
                    state.isConnected ? "bg-green-500" : "bg-red-500"
                  }`}
                />
                <span className="text-gray-400">
                  Status:{" "}
                  <span
                    className={
                      state.isConnected ? "text-green-400" : "text-red-400"
                    }
                  >
                    {state.isConnected ? "Connected" : "Disconnected"}
                  </span>
                </span>
              </div>

              {state.metrics && (
                <div className="flex items-center px-3 py-2">
                  <span className="text-gray-400 mr-2">Threat Level:</span>
                  <span
                    className={`px-2 py-0.5 rounded-md text-sm ${
                      state.metrics.threat_level >= 4
                        ? "bg-red-900/50 text-red-300"
                        : state.metrics.threat_level >= 2
                        ? "bg-yellow-900/50 text-yellow-300"
                        : "bg-green-900/50 text-green-300"
                    }`}
                  >
                    {state.metrics.threat_level}/5
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navigation;
