"""
@file loopback.py
@brief ML-Agents UnityEnvironment whose gRPC server listens on loopback only (thesis section 7.1).

Upstream RpcCommunicator binds "[::]:<port>" (every interface, so a Docker container can connect),
unencrypted and unauthenticated: any host that can reach the port can pose as the trainer or the
simulator. The Unity player connects to localhost, so binding 127.0.0.1 / ::1 loses nothing when
trainer and player share a machine or container, which is how every run here works (local, Slurm).
Running them in separate containers needs an isolated container network and upstream's binding.
"""

from concurrent.futures import ThreadPoolExecutor

import grpc
from mlagents_envs.communicator_objects.unity_to_external_pb2_grpc import (
    add_UnityToExternalProtoServicer_to_server,
)
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.rpc_communicator import RpcCommunicator, UnityToExternalServicerImplementation

## @brief Loopback addresses the server listens on. Both, so "localhost" works whichever one it resolves to;
##        at least one must bind (::1 is absent on hosts with IPv6 disabled).
LOOPBACK_HOSTS = ("127.0.0.1", "[::1]")


class LoopbackRpcCommunicator(RpcCommunicator):
    """@brief RpcCommunicator.create_server, bound to LOOPBACK_HOSTS instead of every interface."""

    def create_server(self):
        self.check_port(self.port)
        try:
            self.server = grpc.server(
                thread_pool=ThreadPoolExecutor(max_workers=10),
                options=(("grpc.so_reuseport", 1),),
            )
            self.unity_to_external = UnityToExternalServicerImplementation()
            add_UnityToExternalProtoServicer_to_server(self.unity_to_external, self.server)
            bound = [host for host in LOOPBACK_HOSTS if _try_bind(self.server, f"{host}:{self.port}")]
            if not bound:
                raise RuntimeError(f"could not bind port {self.port} on {LOOPBACK_HOSTS}")
            self.bound_addresses = [f"{host}:{self.port}" for host in bound]
            self.server.start()
            self.is_open = True
        except Exception:
            raise UnityWorkerInUseException(self.worker_id)


def _try_bind(server, address: str) -> bool:
    try:
        return server.add_insecure_port(address) > 0   # older grpc returns 0 on failure, newer raises
    except RuntimeError:
        return False


class LoopbackUnityEnvironment(UnityEnvironment):
    """@brief UnityEnvironment with a LoopbackRpcCommunicator; otherwise identical."""

    @staticmethod
    def _get_communicator(worker_id, base_port, timeout_wait):
        return LoopbackRpcCommunicator(worker_id, base_port, timeout_wait)
