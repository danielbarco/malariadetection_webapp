# mkdir -p ~/.streamlit

# echo "[server]
# headless = true
# port = $PORT
# enableCORS = false
# " > ~/.streamlit/config.toml

mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"guest@hotmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
enableXsrfProtection=false\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml