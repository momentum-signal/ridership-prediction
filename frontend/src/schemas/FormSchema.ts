import { z } from "zod";

export const FormSchema = z
  .object({
    origin: z.string({
      required_error: "Please select an origin.",
    }),
    destination: z.string({
      required_error: "Please select a destination.",
    }),
    datetime: z.date({
      required_error: "A date and time is required.",
    }),
  })
  .refine((data) => data.origin !== data.destination, {
    message: "Origin and destination cannot be the same.",
    path: ["destination"],
  });
