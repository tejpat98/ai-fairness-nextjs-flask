# Full stack containerised ai-fairness app

## Current State

LIVE DEMO: https://ai-fairness-demo.tejash.dev/

- Only routes for business domain are implemented. 

# Goals:
- Visual UI adjustments
    - Anything need to be removed? Colours? 
- Update placeholder content
    - The website has placeholder text that needs to be double checked and/or replaced.
- Confirm refactored code structure 
    - I have heavily refactored the code from the FDK-toolkit. It is still functionally the same. However, the code is split up differently to improve readability and make it easier for other people to understand.
        - Effectively the data processing of fdk_business.py and fdk_business_pipeline.py are split into:
            - preprocessing.py: feature detection (outputs df_mapped)
            - audit_response() from fdk_business_pipeline.py: for metrics calculations
            - postprocessing(): for creating and saving the reports and summary

## Summary
Currently, the repository https://github.com/AI-Fairness-com/FDK-Toolkit, is a Flask App that uses Jinja templates to handle front end.

The goal of this project is to create a full stack containerised app using this repository that maintains all the same functionality as the original. 

This may require refactoring of the code from the original repository, so that it follows strict RESTful API standards. And there is a clear seperation of concerns between the next.js frontend and Flask API backend.

## Front-end

- Next.js
- v0 / shadcn components
- Nextra for fully featured documentation page

## Back-end

Will use a Flask API, offering the same data processing and analysis capabilities as the original repository, without using Jinja templates for output. 

The container will mount the "./uploads" and "./reports" folders which will be used to store user uploaded datasets for analysis, and reports generated.

Will use waitress as a production ready server rather than "flask run"

## Nginx

Will sit in front of the frontend and backend, routing request accordingly.

Any request to /api/ should be routed to the Flask API container.

All other request should be routed to the next.js frontend container.

## Docker

There will be 4 docker containers: frontend, backend, nginx, worker. Each with a dockerfile in this respective folders (worker shares folder with backend).

There is a primary docker compose file in the root directory.


