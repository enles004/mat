# frontend/Dockerfile — demo UI (Next.js, gọi API qua NEXT_PUBLIC_API_URL).
FROM node:22-alpine AS build
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/app ./app
COPY frontend/public ./public
COPY frontend/tsconfig.json frontend/next.config.ts frontend/next-env.d.ts frontend/eslint.config.mjs ./
ARG NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
RUN npm run build

FROM node:22-alpine AS runtime
WORKDIR /web
ENV NODE_ENV=production
COPY --from=build /web/package.json /web/package-lock.json ./
RUN npm ci --omit=dev
COPY --from=build /web/.next ./.next
COPY --from=build /web/public ./public
COPY --from=build /web/next.config.ts ./next.config.ts
EXPOSE 3000
CMD ["npm", "run", "start", "--", "-p", "3000"]
