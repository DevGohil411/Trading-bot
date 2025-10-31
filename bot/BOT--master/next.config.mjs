/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  async rewrites() {
    const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${API_BASE}/api/:path*`,
      },
      {
        source: '/auth/:path*',
        destination: `${API_BASE}/auth/:path*`,
      },
      {
        source: '/logout',
        destination: `${API_BASE}/logout`,
      },
      {
        source: '/auth/status',
        destination: `${API_BASE}/auth/status`,
      },
    ];
  },
}

export default nextConfig
