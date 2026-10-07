# Hieroglyph Translator (web app)

Installable web app (PWA) that photographs a hieroglyph inscription and will translate it to English.

Live: https://leoleoleovl.github.io/hieroglyphs/ (on iPhone: Share → Add to Home Screen)

## Run locally

```
npm install
npm run dev       # dev server, reachable on your local network
npm run build     # production build in dist/
npm run preview   # serve the build on your local network
```

Pushing changes under `webapp/` to `main` deploys to GitHub Pages via `.github/workflows/deploy-webapp.yml`.
