"""Read-only GitHub infrastructure for job-aware portfolio selection."""

from resume_tailor.backend.github.client import (
    GITHUB_TOKEN_ENV,
    GitHubAPIError,
    GitHubConfigurationError,
    GitHubDossierCache,
    GitHubRepositoryClient,
    GitHubRESTClient,
    default_github_cache_directory,
    discover_repository_snapshots,
    is_safe_repository_path,
    repository_path_exclusion_reason,
)

__all__ = [
    "GITHUB_TOKEN_ENV",
    "GitHubAPIError",
    "GitHubConfigurationError",
    "GitHubDossierCache",
    "GitHubRepositoryClient",
    "GitHubRESTClient",
    "default_github_cache_directory",
    "discover_repository_snapshots",
    "is_safe_repository_path",
    "repository_path_exclusion_reason",
]
