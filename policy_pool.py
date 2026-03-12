class PolicyPool:
    def __init__(self):
        self.policies = []

    def add(self, policy):
        self.policies.append(policy)

    def get_best(self):
        if not self.policies:
            return None
        return self.policies[0]
