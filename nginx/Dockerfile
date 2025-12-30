# Use the official Nginx image
FROM nginx:alpine

#Remove old config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom nginx configuration
COPY default.conf /etc/nginx/conf.d/default.conf
