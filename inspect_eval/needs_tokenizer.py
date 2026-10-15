"""Skip tests that need Inspect's tokenizer when it can be neither found nor downloaded.

Before every turn the task asks Inspect to estimate the prompt's tokens (the context budget), and Inspect
does that with tiktoken's o200k_base, a 3.6 MB file it downloads the first time it is used. On a machine
with no network, or one that blocks the download, every test that runs the task would fail for that
reason alone. Those tests are decorated with ``needs_tokenizer`` and are reported as skipped, with the
reason, instead. Run the suite once with network access and the file is cached.
"""
import socket
import unittest


def _tokenizer_loads() -> bool:
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(10)  # a blocked download should end the check, not hang it
    try:
        import tiktoken

        tiktoken.get_encoding("o200k_base")
        return True
    except Exception:
        return False
    finally:
        socket.setdefaulttimeout(previous)


needs_tokenizer = unittest.skipUnless(
    _tokenizer_loads(),
    "needs tiktoken's o200k_base file (3.6 MB): run the tests once with network access to cache it",
)
