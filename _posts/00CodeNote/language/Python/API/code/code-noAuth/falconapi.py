import json

import falcon


class CompaniesResource:
    companies = [{"id": 1, "name": "Company One"}, {"id": 2, "name": "Company Two"}]

    def on_get(self, req, resp):
        resp.body = json.dumps(self.companies)

    def on_post(self, req, resp):
        resp.status = falcon.HTTP_201
        resp.body = json.dumps({"success": True})


api = falcon.API()
companies_endpoint = CompaniesResource()

api.add_route("/companies", companies_endpoint)
