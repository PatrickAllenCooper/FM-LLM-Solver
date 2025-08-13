import { Outlet, Link, useLocation } from 'react-router-dom';
import { Fragment } from 'react';
import { Disclosure, Menu, Transition } from '@headlessui/react';
import {
  Bars3Icon,
  XMarkIcon,
  BeakerIcon,
  DocumentTextIcon,
  CpuChipIcon,
  ChartBarIcon,
  UserIcon,
  ArrowRightOnRectangleIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';
import { useAuthStore } from '@/stores/auth.store';
import { clsx } from 'clsx';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: ChartBarIcon },
  { name: 'System Specs', href: '/system-specs', icon: DocumentTextIcon },
  { name: 'Certificates', href: '/certificates', icon: CpuChipIcon },
  { name: 'About', href: '/about', icon: InformationCircleIcon },
  { name: 'Experiments', href: '/experiments', icon: BeakerIcon },
];

export default function Layout() {
  const location = useLocation();
  const { user, logout } = useAuthStore();

  return (
    <div className="min-h-full">
      <Disclosure as="nav" className="bg-white shadow-md border-b border-gray-100">
        {({ open }) => (
          <>
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
              <div className="flex h-20 justify-between">
                <div className="flex">
                  <div className="flex flex-shrink-0 items-center">
                    <Link to="/dashboard" className="flex items-center">
                      <CpuChipIcon className="h-8 w-8 text-primary-600" />
                      <span className="ml-2 text-xl font-bold text-gradient">
                        FM-LLM Solver
                      </span>
                    </Link>
                  </div>
                  <div className="hidden sm:ml-8 sm:flex sm:space-x-2">
                    {navigation.map((item) => {
                      const isActive = location.pathname.startsWith(item.href);
                      return (
                        <Link
                          key={item.name}
                          to={item.href}
                          className={clsx(
                            isActive
                              ? 'bg-primary-100 text-primary-700 shadow-sm'
                              : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50',
                            'inline-flex items-center px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 hover:shadow-md'
                          )}
                        >
                          <item.icon className="mr-2 h-4 w-4" />
                          {item.name}
                        </Link>
                      );
                    })}
                  </div>
                </div>
                <div className="hidden sm:ml-6 sm:flex sm:items-center">
                  <Menu as="div" className="relative ml-3">
                    <div>
                      <Menu.Button className="flex max-w-xs items-center rounded-full bg-white px-3 py-2 text-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-all duration-200 shadow-sm hover:shadow-md">
                        <span className="sr-only">Open user menu</span>
                        <div className="h-8 w-8 rounded-full bg-gradient-to-br from-cu-gold to-primary-600 flex items-center justify-center shadow-sm">
                          <UserIcon className="h-4 w-4 text-cu-black" />
                        </div>
                        <span className="ml-3 text-sm font-medium text-gray-800">
                          {user?.email}
                        </span>
                      </Menu.Button>
                    </div>
                    <Transition
                      as={Fragment}
                      enter="transition ease-out duration-200"
                      enterFrom="transform opacity-0 scale-95"
                      enterTo="transform opacity-100 scale-100"
                      leave="transition ease-in duration-75"
                      leaveFrom="transform opacity-100 scale-100"
                      leaveTo="transform opacity-0 scale-95"
                    >
                      <Menu.Items className="absolute right-0 z-10 mt-2 w-56 origin-top-right rounded-2xl bg-white py-2 shadow-xl ring-1 ring-gray-200 focus:outline-none">
                        <Menu.Item>
                          {({ active }) => (
                            <Link
                              to="/profile"
                              className={clsx(
                                active ? 'bg-primary-50 text-primary-700' : 'text-gray-700',
                                'flex items-center px-4 py-3 text-sm font-medium rounded-xl mx-2 transition-colors duration-150'
                              )}
                            >
                              <UserIcon className="mr-3 h-4 w-4" />
                              Profile
                            </Link>
                          )}
                        </Menu.Item>
                        <Menu.Item>
                          {({ active }) => (
                            <button
                              onClick={logout}
                              className={clsx(
                                active ? 'bg-red-50 text-red-700' : 'text-gray-700',
                                'flex w-full items-center px-4 py-3 text-sm font-medium rounded-xl mx-2 transition-colors duration-150'
                              )}
                            >
                              <ArrowRightOnRectangleIcon className="mr-3 h-4 w-4" />
                              Sign out
                            </button>
                          )}
                        </Menu.Item>
                      </Menu.Items>
                    </Transition>
                  </Menu>
                </div>
                <div className="-mr-2 flex items-center sm:hidden">
                  <Disclosure.Button className="inline-flex items-center justify-center rounded-full bg-white p-3 text-gray-400 hover:bg-gray-100 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 shadow-sm hover:shadow-md transition-all duration-200">
                    <span className="sr-only">Open main menu</span>
                    {open ? (
                      <XMarkIcon className="block h-6 w-6" aria-hidden="true" />
                    ) : (
                      <Bars3Icon className="block h-6 w-6" aria-hidden="true" />
                    )}
                  </Disclosure.Button>
                </div>
              </div>
            </div>

            <Disclosure.Panel className="sm:hidden bg-white border-t border-gray-100">
              <div className="space-y-2 p-4">
                {navigation.map((item) => {
                  const isActive = location.pathname.startsWith(item.href);
                  return (
                    <Disclosure.Button
                      key={item.name}
                      as={Link}
                      to={item.href}
                      className={clsx(
                        isActive
                          ? 'bg-primary-100 text-primary-700 shadow-sm'
                          : 'text-gray-600 hover:bg-gray-50 hover:text-gray-800',
                        'block py-3 px-4 text-base font-medium rounded-xl transition-all duration-200'
                      )}
                    >
                      <div className="flex items-center">
                        <item.icon className="mr-3 h-5 w-5" />
                        {item.name}
                      </div>
                    </Disclosure.Button>
                  );
                })}
              </div>
              <div className="border-t border-gray-100 pb-4 pt-4">
                <div className="flex items-center px-4">
                  <div className="h-10 w-10 rounded-full bg-gradient-to-br from-cu-gold to-primary-600 flex items-center justify-center shadow-sm">
                    <UserIcon className="h-5 w-5 text-cu-black" />
                  </div>
                  <div className="ml-3">
                    <div className="text-base font-medium text-gray-800">{user?.email}</div>
                    <div className="text-sm font-medium text-gray-500 capitalize">{user?.role}</div>
                  </div>
                </div>
                <div className="mt-4 space-y-2 px-4">
                  <Disclosure.Button
                    as={Link}
                    to="/profile"
                    className="block px-4 py-3 text-base font-medium text-gray-600 hover:bg-primary-50 hover:text-primary-700 rounded-xl transition-colors duration-150"
                  >
                    Profile
                  </Disclosure.Button>
                  <Disclosure.Button
                    as="button"
                    onClick={logout}
                    className="block w-full text-left px-4 py-3 text-base font-medium text-gray-600 hover:bg-red-50 hover:text-red-700 rounded-xl transition-colors duration-150"
                  >
                    Sign out
                  </Disclosure.Button>
                </div>
              </div>
            </Disclosure.Panel>
          </>
        )}
      </Disclosure>

      <div className="min-h-screen bg-gray-50">
        <main className="py-8">
          <div className="mx-auto max-w-7xl px-6 lg:px-8">
            <Outlet />
          </div>
        </main>
        
        {/* CU Boulder Footer */}
        <footer className="border-t border-gray-200 bg-white">
          <div className="mx-auto max-w-7xl px-6 lg:px-8 py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 rounded-full cu-gradient flex items-center justify-center">
                  <CpuChipIcon className="h-4 w-4 text-cu-black" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900">FM-LLM Solver</p>
                  <p className="text-xs text-gray-500">University of Colorado Boulder</p>
                </div>
              </div>
              <div className="text-xs text-gray-500">
                Rigorous evaluation of LLMs for formal verification
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
