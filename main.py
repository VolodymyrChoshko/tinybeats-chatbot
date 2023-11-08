from server import Backend
from server import app
from server import config
# from server.dbclient import supabase
from server.dbclient.pgvector import CONNECTION_STRING
from server import celery_init
from utils import celery_functions

app = config(app)
celery = celery_init(app)
_calculate_embeddings = celery_functions(celery)

# backend  = Backend(app, supabase, _calculate_embeddings)
backend  = Backend(app, CONNECTION_STRING, _calculate_embeddings)


for route in backend.routes:
    app.add_url_rule(
        route,
        view_func = backend.routes[route]['function'],
        methods   = backend.routes[route]['methods'],
    )


if __name__ == '__main__':
    app.run(port=8080, host='0.0.0.0')