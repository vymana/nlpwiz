
from nameko.rpc import rpc

from nlpwiz.server import model_factory

class ModelsService:
    name = "model_service"
    @rpc
    def run(self, method_name, kwargs={}):
        method = getattr(model_factory, method_name)
        return method(**kwargs)

    @rpc
    def test(self, model_name, *args, **kwargs):
        #with ClusterRpcProxy(config) as cluster_rpc:
        #    return cluster_rpc.model_service.infer(model_name, *args, **kwargs)
        return "test method"

