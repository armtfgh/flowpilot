import { expect, test } from "@playwright/test";


async function expectNoPageOverflow(page: import("@playwright/test").Page) {
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth + 1);
}


test("populated design uses one canonical result across tabs", async ({ page }, testInfo) => {
  await page.goto("/?demo=1");
  await expect(page.getByRole("heading", { name: "Design studio" })).toBeVisible();
  await expect(page.getByText("Executable screening design")).toBeVisible();
  await expect(page.getByText("8.05 min", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Process" }).click();
  await expect(page.getByRole("heading", { name: "Executable process topology" })).toBeVisible();
  await expect(page.locator(".processNode")).toHaveCount(7);
  await expect(page.getByText("rx_1", { exact: true }).first()).toBeVisible();

  await page.getByRole("button", { name: "Engineering" }).click();
  await expect(page.getByText("ST-01", { exact: true })).toBeVisible();
  await expect(page.getByText("8.05 / 8.05 min", { exact: true })).toBeVisible();
  await expect(page.getByText("substrate; photocatalyst in MeCN", { exact: true })).toBeVisible();
  await expectNoPageOverflow(page);

  await page.screenshot({
    path: testInfo.outputPath("flowpilot-design.png"),
    fullPage: true,
  });
});


test("inventory workspace and mobile navigation remain usable", async ({ page }, testInfo) => {
  await page.goto("/?demo=1");
  const isMobile = testInfo.project.name === "mobile";
  if (isMobile) {
    await page.getByRole("button", { name: "Menu" }).click();
  }
  await page.getByRole("button", { name: /Inventory/ }).click();
  await expect(page.getByRole("heading", { name: "Import laboratory inventory" })).toBeVisible();
  await expect(page.getByText("Choose PDF, Word, PowerPoint, spreadsheet, or JSON")).toBeVisible();
  await expectNoPageOverflow(page);
  await page.screenshot({
    path: testInfo.outputPath("flowpilot-inventory.png"),
    fullPage: true,
  });
});


test("mandatory chemistry intake closes and model routes are selectable", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Initial batch protocol").fill(
    "Compound A (1.0 mmol) was treated with reagent B (1.2 mmol) in acetonitrile at 40 C for 2 h and afforded compound C."
  );
  const llmExtraction = page.locator(".intakePanel .toggle input");
  await llmExtraction.uncheck({ force: true });
  await expect(llmExtraction).not.toBeChecked();
  await page.getByRole("button", { name: "Analyze intake" }).click();

  const chemistry = page.locator(".question").filter({ hasText: "Q-CHEM-001" });
  await expect(chemistry).toBeVisible();
  await expect(chemistry.getByText("Explicitly unavailable")).toHaveCount(0);
  await chemistry.locator("textarea").fill(
    "Ring closure of precursor A gives cyclic product C."
  );

  const objective = page.locator(".question").filter({ hasText: "Q-OBJ-001" });
  await objective.locator("textarea").fill("Produce a conservative first flow screen.");
  for (const questionId of ["Q-HIST-001", "Q-INV-001", "Q-CONSTR-001", "Q-HYP-001"]) {
    const question = page.locator(".question").filter({ hasText: questionId });
    if (await question.count()) await question.getByText("Explicitly unavailable").click();
  }
  await page.getByRole("button", { name: "Save answers" }).click();
  await expect(page.getByRole("heading", { name: "Design input is frozen" })).toBeVisible();
  await expect(page.getByRole("button", { name: /Run FlowPilot design/ })).toBeEnabled();

  const upstream = page.getByLabel("Upstream chemistry model");
  const downstream = page.getByLabel("Downstream and council model");
  await expect(upstream.locator("option")).toHaveCount(5);
  for (const select of [upstream, downstream]) {
    expect(await select.locator('option[value="gpt-4o"]').evaluate(
      (option: HTMLOptionElement) => option.disabled
    )).toBe(true);
    expect(await select.locator('option[value="qwen3.8-27b"]').evaluate(
      (option: HTMLOptionElement) => option.disabled
    )).toBe(true);
  }
  await upstream.selectOption("claude-opus-4-6");
  await downstream.selectOption("qwen3.6-27b");
  await expect(upstream).toHaveValue("claude-opus-4-6");
  await expect(downstream).toHaveValue("qwen3.6-27b");
});
