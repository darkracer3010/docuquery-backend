from supabase import Client, AuthApiError


class AuthService:
    def __init__(self, supabase_client: Client):
        self.client = supabase_client

    def sign_up(self, email: str, password: str, full_name: str | None = None) -> dict:
        """Register a new user via Supabase Auth."""
        try:
            response = self.client.auth.sign_up(
                {
                    "email": email,
                    "password": password,
                    "options": {
                        "data": {"full_name": full_name} if full_name else {},
                    },
                }
            )

            if response.user:
                # Create profile row
                self.client.table("profiles").insert(
                    {
                        "id": response.user.id,
                        "full_name": full_name,
                    }
                ).execute()

            return {
                "user_id": response.user.id if response.user else None,
                "email": response.user.email if response.user else None,
                "access_token": response.session.access_token if response.session else None,
                "refresh_token": response.session.refresh_token if response.session else None,
            }
        except AuthApiError as e:
            raise ValueError(str(e))

    def sign_in(self, email: str, password: str) -> dict:
        """Sign in with email/password."""
        try:
            response = self.client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            return {
                "user_id": response.user.id,
                "email": response.user.email,
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token,
            }
        except AuthApiError as e:
            raise ValueError(str(e))

    def get_profile(self, user_id: str) -> dict:
        """Get user profile by ID."""
        response = (
            self.client.table("profiles")
            .select("*")
            .eq("id", user_id)
            .single()
            .execute()
        )
        return response.data
