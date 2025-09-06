/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,

  typescript: {
    ignoreBuildErrors: false,
  },
  eslint: {
    ignoreDuringBuilds: false,
  },
  // Remove output: "standalone" for Render deployment
  images: {
    domains: [],
    unoptimized: true,
  },

  webpack: (config, { isServer }) => {
    // Handle audio files
    config.module.rules.push({
      test: /\.(mp3|wav|ogg|m4a)$/,
      use: {
        loader: "file-loader",
        options: {
          publicPath: "/_next/static/audio/",
          outputPath: "static/audio/",
        },
      },
    })

    // Handle Python files (for scripts)
    config.module.rules.push({
      test: /\.py$/,
      use: "raw-loader",
    })

    return config
  },
}

module.exports = nextConfig
