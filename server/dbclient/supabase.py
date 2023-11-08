from supabase import create_client, Client
import os

supabase: Client = create_client(os.environ.get('SUPABASE_URL'), os.environ.get('SUPABASE_API_KEY'))