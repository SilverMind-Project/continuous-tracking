FROM --platform=$BUILDPLATFORM golang:1.26.3-bookworm AS builder
ARG TARGETOS
ARG TARGETARCH
WORKDIR /src/rtsp-ingress
COPY proto /src/proto
COPY rtsp-ingress/go.mod rtsp-ingress/go.sum ./
RUN go mod download
COPY rtsp-ingress /src/rtsp-ingress
RUN GOOS=$TARGETOS GOARCH=$TARGETARCH CGO_ENABLED=0 go build -o /out/rtsp-ingress ./cmd/server

FROM debian:bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /out/rtsp-ingress /app/rtsp-ingress
COPY rtsp-ingress/config /app/config
USER 65532:65532
EXPOSE 8090
CMD ["/app/rtsp-ingress"]
