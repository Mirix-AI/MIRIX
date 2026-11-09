# Provider credential exposure: owner follow-up

This change addresses the Gemini/OpenAI credential exposure only.

The Gemini REST path in `agent.py` contained a provider key directly in source.
It now requires `GEMINI_API_KEY`, fails before making a request when the key is
missing, and sends the configured key in `x-goog-api-key` rather than the URL.
Set a **newly issued** key in the environment or an untracked `.env` file.

As checked on 2026-11-09, the credential was present in both branch heads:

- `public_evaluation`: `33e4bde55b005508111bd9f354b74050a0da0c7b` (this change's base).
- `evals`: `5def761103ca1c6f8e35a820abfc0eec4edd4f5f` (apply the same fix there).

An OpenAI project key was also committed in the historical
`streaming-demo/server.js` and `streaming-demo/server.py` files. These files are
absent from the checked branch heads, so there is no current fallback key to
remove from them.

For Yu Wang, as the credential owner:

1. Revoke the previously exposed Gemini and OpenAI keys in their respective
   provider accounts. If already revoked, confirm that status.
2. Update applications using those credentials with newly issued replacements.
3. Review provider usage and billing for unexpected activity.

Removing a literal from the latest code does **not** revoke it or remove it from
existing Git history and PR references. This commit does not claim revocation.
No exposed key was used to contact either provider during this verification.
The key values are intentionally omitted from this note.

Offline regression checks:

```sh
python3 -m unittest discover -s public_evaluations/tests -p test_provider_credentials.py -v
```

The tests execute the actual Gemini request method with mocked HTTP, verify the
configured header and missing-key failure, and reject embedded Gemini/OpenAI key
literals in the evaluation client.
