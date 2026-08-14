# Stage 1: build the static assets. Needs Node only to run the build -
# none of this (npm, node_modules) ends up in the final image.
FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: serve the built files with nginx. Final image is just nginx +
# static HTML/JS/CSS - no Node runtime, no source code, much smaller
# than shipping the build stage itself.
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80