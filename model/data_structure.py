class Results:

    def __init__(self, parameters, indirect_exchanges, direct_exchanges):
        self.indirect_exchanges = indirect_exchanges
        self.direct_exchanges = direct_exchanges
        self.parameters = parameters


class PoolResults:

    def __init__(self, parameters, results):
        self.parameters = parameters
        self.results = results

