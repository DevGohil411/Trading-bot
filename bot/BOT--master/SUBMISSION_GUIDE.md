# Submission Guide (BOT--master)

Include these files/folders in your submission:
- `README.md` (setup and run)
- `ENV.example` (copy to `.env.local`)
- `DEPENDENCIES.md`
- `app/`, `components/`, `hooks/`, `lib/`, `pages/`, `styles/`, `tailwind.config.ts`, `tsconfig.json`, `next.config.mjs`, `package.json`, `package-lock.json`
- Optional backend: `server/` + `server/env.example`

Quick run:
1) `npm install`
2) `npm run dev` (http://localhost:3000)

Notes:
- The UI expects Python API at `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).
- Do not commit secrets; use `.env.local` locally. 