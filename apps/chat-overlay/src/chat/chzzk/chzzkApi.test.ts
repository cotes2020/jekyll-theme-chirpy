import { describe, expect, it } from "vitest";

import { parseSessionUrlFromAuthResponse } from "./chzzkApi";

describe("parseSessionUrlFromAuthResponse", () => {
  it("content.url을 반환한다", () => {
    expect(
      parseSessionUrlFromAuthResponse({
        code: 200,
        message: null,
        content: { url: " https://ssio.example.com/ " }
      })
    ).toBe("https://ssio.example.com/");
  });

  it("code가 200이 아니면 예외", () => {
    expect(() =>
      parseSessionUrlFromAuthResponse({ code: 401, message: "bad", content: {} })
    ).toThrow();
  });
});
