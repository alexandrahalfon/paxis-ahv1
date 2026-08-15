"""
Community subsystem (Phase 7).

Structurally and physically separate from clinical PHI and from the
evidence-retrieval layer — see the architecture review sections 32-34.
Nothing in this package is ever merged into a retrieval ranking pool or
used as clinical evidence; a community post answers "what did other
people find helpful", never "this treatment is safe".

Identity here is a pseudonymous community_profile (handle + optional
high-level tags), never the patient's real name, diagnosis detail, or
clinical facts. community_profiles.user_id is the only link back to a
real account, and it exists solely for auth/moderation — content-facing
reads never expose it, only the handle.
"""
