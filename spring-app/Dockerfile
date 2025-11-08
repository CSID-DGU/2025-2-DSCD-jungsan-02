FROM gradle:8.5-jdk21 AS builder
WORKDIR /app
COPY . /app
RUN gradle wrapper --gradle-version 8.5
RUN ./gradlew clean build -x test

FROM eclipse-temurin:21-jre
COPY --from=builder /app/build/libs/lostfound-0.0.1-SNAPSHOT.jar /lostfound.jar
EXPOSE 8080
CMD ["java", "-jar", "/lostfound.jar"]