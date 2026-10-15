import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Relative base: the same build works on GitHub Pages (/squadron-zero/) and from a local folder offline.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { chunkSizeWarningLimit: 2000 },
});
