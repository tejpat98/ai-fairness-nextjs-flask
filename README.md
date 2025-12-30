# Full stack containerised ai-fairness app

## Summary
Currently, the repository https://github.com/AI-Fairness-com/FDK-Toolkit, is a Flask App that uses Jinja templates to handle front end.

The goal of this project is to create a full stack containerised app using this repository that maintains all the same functionality as the original. 

This may require refactoring of the code from the original repository, so that it follows strict RESTful API standards. And there is a clear seperation of concerns between the next.js frontend and Flask API backend.

## Front-end

Will use Next.js

## Back-end

Will use a Flask API, offering the same data processing and analysis capabilities as the original repository, without using Jinja templates for output. 

The container will mount the "./data" folder which will be used to store user uploaded datasets for analysis, and reports generated.

## Nginx

Will sit in front of the frontend and backend, routing request accordingly.

Any request to /api/ should be routed to the Flask API container.

All other request should be routed to the next.js frontend container.

## Docker

There will be 3 docker containers: frontend, backend, nginx. Each with a dockerfile in this respective folders.

There is a primary docker compose file in the root directory.

