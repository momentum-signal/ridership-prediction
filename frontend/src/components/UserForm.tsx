"use client"

import { zodResolver } from "@hookform/resolvers/zod"
import { Check, ChevronsUpDown, ArrowLeftRight } from "lucide-react"
import { useForm } from "react-hook-form"
import { toast } from "sonner"
import { z } from "zod"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { DateTimePicker } from '@/components/ui/datetime-picker';
const stations = [
  { label: "Kezang", value: "kezang" },
  { label: "Phelio Damansara", value: "Phelio Damansara" },
  { label: "Puchong", value: "puchong" },
  { label: "Kuala Lumpur", value: "kuala-lumpur" },
  { label: "Petaling Jaya", value: "petaling-jaya" },
  { label: "Subang Jaya", value: "subang-jaya" },
  { label: "Bandar Utama", value: "bandar-utama" },
  { label: "Bangsar", value: "bangsar" },
  { label: "Titiwangsa", value: "titiwangsa" },
  { label: "Sentul", value: "sentul" },
] as const

const FormSchema = z.object({
  origin: z.string({
    required_error: "Please select an origin.",
  }),
  destination: z.string({
    required_error: "Please select a destination.",
  }),
  datetime: z.date({
    required_error: "A date and time is required.",
  }),
}).refine((data) => data.origin !== data.destination, {
  message: "Origin and destination cannot be the same.",
  path: ["destination"],
});
const DEFAULT_VALUE = {
  datetime: new Date(),
};

export default function UserForm() {
  const form = useForm<z.infer<typeof FormSchema>>({
    defaultValues: DEFAULT_VALUE,
    resolver: zodResolver(FormSchema),
  })
  function onSubmit(data: z.infer<typeof FormSchema>) {
    toast("You submitted the following values", {
      description: (
        <pre className="mt-2 w-[320px] rounded-md bg-neutral-950 p-4">
          <code className="text-white">{JSON.stringify(data, null, 2)}</code>
        </pre>
      ),
    })
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="flex flex-col gap-6 w-full">
        {/* Origin & Destination with Swap */}        <div className="flex flex-col gap-2 w-full">
          <div className="flex flex-col gap-1 bg-card/70 rounded-xl p-4 shadow-sm border border-border">
            <FormLabel className="text-base font-semibold">Origin <span className="text-destructive">*</span></FormLabel>
            <FormField
              control={form.control}
              name="origin"
              render={({ field }) => (
                <FormItem className="flex flex-col gap-1">
                  <Popover>
                    <PopoverTrigger asChild>
                      <FormControl>
                        <Button
                          variant="outline"
                          role="combobox"
                          className={cn(
                            "w-full justify-between text-left font-medium border-2 focus:ring-2 focus:ring-primary/50 h-12 rounded-lg bg-background/80",
                            !field.value && "text-muted-foreground"
                          )}
                        >
                          {field.value
                            ? stations.find((station) => station.value === field.value)?.label
                            : "Select Origin"}
                          <ChevronsUpDown className="opacity-50 ml-2" />
                        </Button>
                      </FormControl>
                    </PopoverTrigger>
                    <PopoverContent align="start" sideOffset={4} className="w-[var(--radix-popover-trigger-width)] min-w-0 max-w-full p-0">
                      <Command>
                        <CommandInput placeholder="Search Origin..." className="h-9" />
                        <CommandList>
                          <CommandEmpty>No station found.</CommandEmpty>
                          <CommandGroup>
                            {stations.map((station) => (
                              <CommandItem
                                value={station.label}
                                key={station.value}
                                onSelect={() => {
                                  form.setValue("origin", station.value, { shouldValidate: true });
                                }}
                                disabled={form.watch("destination") === station.value}
                              >
                                {station.label}
                                <Check className={cn("ml-auto", station.value === field.value ? "opacity-100" : "opacity-0")} />
                              </CommandItem>
                            ))}
                          </CommandGroup>
                        </CommandList>
                      </Command>
                    </PopoverContent>
                  </Popover>
                  <FormMessage />
                </FormItem>
              )}
            />          </div>
          {/* Swap Button */}
          <div className="flex justify-center items-center -my-2">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label="Swap origin and destination"
              className="border border-border rounded-full shadow-md bg-background hover:bg-accent focus:ring-2 focus:ring-primary mx-0 my-0 p-3"
              onClick={() => {
                const origin = form.getValues("origin");
                const destination = form.getValues("destination");
                form.setValue("origin", destination, { shouldValidate: true });
                form.setValue("destination", origin, { shouldValidate: true });
              }}
            >
              <ArrowLeftRight className="h-6 w-6 transition-transform duration-200 sm:rotate-0 rotate-90 text-primary" />            </Button>
          </div>          <div className="flex flex-col gap-1 bg-card/70 rounded-xl p-4 shadow-sm border border-border">
            <FormLabel className="text-base font-semibold">Destination <span className="text-destructive">*</span></FormLabel>
            <FormField
              control={form.control}
              name="destination"
              render={({ field }) => (
                <FormItem className="flex flex-col gap-1">
                  <Popover>
                    <PopoverTrigger asChild>
                      <FormControl>
                        <Button
                          variant="outline"
                          role="combobox"
                          className={cn(
                            "w-full justify-between text-left font-medium border-2 focus:ring-2 focus:ring-primary/50 h-12 rounded-lg bg-background/80",
                            !field.value && "text-muted-foreground"
                          )}
                        >
                          {field.value
                            ? stations.find((station) => station.value === field.value)?.label
                            : "Select destination"}
                          <ChevronsUpDown className="opacity-50 ml-2" />
                        </Button>
                      </FormControl>
                    </PopoverTrigger>
                    <PopoverContent align="start" sideOffset={4} className="w-[var(--radix-popover-trigger-width)] min-w-0 max-w-full p-0">
                      <Command>
                        <CommandInput placeholder="Search Destination..." className="h-9" />
                        <CommandList>
                          <CommandEmpty>No station found.</CommandEmpty>
                          <CommandGroup>
                            {stations.map((station) => (
                              <CommandItem
                                value={station.label}
                                key={station.value}
                                onSelect={() => {
                                  form.setValue("destination", station.value, { shouldValidate: true });
                                }}
                                disabled={form.watch("origin") === station.value}
                              >
                                {station.label}
                                <Check className={cn("ml-auto", station.value === field.value ? "opacity-100" : "opacity-0")} />
                              </CommandItem>
                            ))}
                          </CommandGroup>
                        </CommandList>
                      </Command>
                    </PopoverContent>
                  </Popover>
                  <FormMessage />
                </FormItem>
              )}
            />          </div>
        </div>        {/* DateTime Picker */}
        <div className="w-full flex flex-col gap-1 bg-card/70 rounded-xl p-4 shadow-sm border border-border">
          <FormLabel className="text-base font-semibold" htmlFor="datetime">Date time <span className="text-destructive">*</span></FormLabel>
          <FormField
            control={form.control}
            name="datetime" render={({ field }) => (
              <FormItem className="flex flex-col gap-1 w-full">
                <FormControl>
                  <DateTimePicker
                    value={field.value}
                    onChange={field.onChange}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>
        <Button type="submit" className="w-full py-3 text-base font-semibold rounded-lg mt-2">Search</Button>
      </form>
    </Form>
  )
}
