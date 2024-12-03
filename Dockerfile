FROM ubuntu:latest
LABEL authors="Michel"

ENTRYPOINT ["top", "-b"]