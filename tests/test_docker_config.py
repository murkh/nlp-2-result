"""
Unit and Integration Tests for Docker and Docker Compose Configurations.
Validates Dockerfile.backend, Dockerfile.frontend, and docker-compose.yml structure,
service definitions, healthchecks, dependencies, ports, and volumes.
"""

import re
import unittest
from pathlib import Path

try:
    import yaml

    _has_yaml = True
except ImportError:
    _has_yaml = False


class TestDockerConfigurations(unittest.TestCase):
    """Test suite for Docker and Docker Compose deployment configs."""

    def setUp(self):
        self.root_dir = Path(__file__).parent.parent
        self.backend_dockerfile = self.root_dir / "Dockerfile.backend"
        self.frontend_dockerfile = self.root_dir / "Dockerfile.frontend"
        self.docker_compose_file = self.root_dir / "docker-compose.yml"

    # -------------------------------------------------------------------------
    # Dockerfile.backend Tests
    # -------------------------------------------------------------------------

    def test_dockerfile_backend_exists_and_non_empty(self):
        """Verify Dockerfile.backend exists and is populated."""
        self.assertTrue(self.backend_dockerfile.exists(), "Dockerfile.backend must exist")
        content = self.backend_dockerfile.read_text().strip()
        self.assertGreater(len(content), 50, "Dockerfile.backend must not be empty")

    def test_dockerfile_backend_directives(self):
        """Verify Dockerfile.backend contains valid base image, packages, and healthcheck."""
        content = self.backend_dockerfile.read_text()

        # Base image
        self.assertRegex(
            content,
            r"FROM\s+python:3\.11-slim",
            "Dockerfile.backend should use python:3.11-slim base",
        )

        # Workdir
        self.assertIn("WORKDIR /app", content)

        # System dependencies
        self.assertIn("libpq-dev", content)
        self.assertIn("curl", content)

        # UV package manager
        self.assertIn("uv", content)

        # Expose port
        self.assertIn("EXPOSE 8000", content)

        # Healthcheck
        self.assertIn("HEALTHCHECK", content)
        self.assertIn("http://localhost:8000/health", content)

        # CMD entrypoint
        self.assertIn("uvicorn", content)
        self.assertIn("src.main:app", content)

    # -------------------------------------------------------------------------
    # Dockerfile.frontend Tests
    # -------------------------------------------------------------------------

    def test_dockerfile_frontend_exists_and_non_empty(self):
        """Verify Dockerfile.frontend exists and is populated."""
        self.assertTrue(self.frontend_dockerfile.exists(), "Dockerfile.frontend must exist")
        content = self.frontend_dockerfile.read_text().strip()
        self.assertGreater(len(content), 50, "Dockerfile.frontend must not be empty")

    def test_dockerfile_frontend_directives(self):
        """Verify Dockerfile.frontend contains valid Streamlit base and healthcheck."""
        content = self.frontend_dockerfile.read_text()

        # Base image & workdir
        self.assertRegex(
            content,
            r"FROM\s+python:3\.11-slim",
            "Dockerfile.frontend should use python:3.11-slim base",
        )
        self.assertIn("WORKDIR /app", content)

        # Dependencies
        self.assertIn("streamlit", content)
        self.assertIn("httpx", content)
        self.assertIn("pandas", content)

        # Expose port
        self.assertIn("EXPOSE 8501", content)

        # Healthcheck
        self.assertIn("HEALTHCHECK", content)
        self.assertIn("8501", content)

        # CMD
        self.assertIn("streamlit", content)
        self.assertIn("frontend/ui.py", content)

    # -------------------------------------------------------------------------
    # docker-compose.yml Tests
    # -------------------------------------------------------------------------

    def test_docker_compose_file_exists(self):
        """Verify docker-compose.yml exists in repository root."""
        self.assertTrue(self.docker_compose_file.exists(), "docker-compose.yml must exist")

    def test_docker_compose_services_and_structure(self):
        """Verify docker-compose.yml defines 4 required services and volume mounts."""
        content = self.docker_compose_file.read_text()

        # Check required services exist
        required_services = ["postgres", "backend", "frontend", "langfuse"]
        for svc in required_services:
            self.assertIn(f"{svc}:", content, f"docker-compose.yml must define '{svc}' service")

        # Check port exposures
        self.assertIn('"5432:5432"', content)
        self.assertIn('"8000:8000"', content)
        self.assertIn('"8501:8501"', content)
        self.assertIn('"3000:3000"', content)

        # Check pgvector image
        self.assertIn("pgvector/pgvector:pg16", content)

        # Check langfuse image
        self.assertIn("langfuse/langfuse:2", content)

        # Check healthchecks and dependencies
        self.assertIn("pg_isready", content)
        self.assertIn("service_healthy", content)
        self.assertIn("postgres_data", content)

    def test_docker_compose_yaml_parsing_if_available(self):
        """Verify docker-compose.yml parses as valid YAML if PyYAML is installed."""
        if not _has_yaml:
            self.skipTest("PyYAML not installed")

        content = self.docker_compose_file.read_text()
        data = yaml.safe_load(content)

        self.assertIsInstance(data, dict)
        self.assertIn("services", data)
        services = data["services"]

        # 4 Core Services
        self.assertIn("postgres", services)
        self.assertIn("backend", services)
        self.assertIn("frontend", services)
        self.assertIn("langfuse", services)

        # Postgres config
        postgres_svc = services["postgres"]
        self.assertEqual(postgres_svc.get("image"), "pgvector/pgvector:pg16")
        self.assertIn("healthcheck", postgres_svc)

        # Backend config
        backend_svc = services["backend"]
        self.assertIn("depends_on", backend_svc)
        self.assertIn("postgres", backend_svc["depends_on"])

        # Frontend config
        frontend_svc = services["frontend"]
        self.assertIn("depends_on", frontend_svc)

        # Langfuse config
        langfuse_svc = services["langfuse"]
        self.assertEqual(langfuse_svc.get("image"), "langfuse/langfuse:2")

        # Volumes
        self.assertIn("volumes", data)
        self.assertIn("postgres_data", data["volumes"])

    def test_dockerfile_multi_stage_builds(self):
        """Verify Dockerfile.backend and Dockerfile.frontend define multi-stage builder and runner."""
        backend_content = self.backend_dockerfile.read_text()
        frontend_content = self.frontend_dockerfile.read_text()

        self.assertIn("AS builder", backend_content)
        self.assertIn("AS runner", backend_content)
        self.assertIn("COPY --from=builder", backend_content)

        self.assertIn("AS builder", frontend_content)
        self.assertIn("AS runner", frontend_content)
        self.assertIn("COPY --from=builder", frontend_content)

    def test_docker_compose_networks(self):
        """Verify docker-compose defines custom bridge network."""
        content = self.docker_compose_file.read_text()
        self.assertIn("nlp_network", content)
        self.assertIn("networks:", content)


if __name__ == "__main__":
    unittest.main()
