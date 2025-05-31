"use client"

import { zodResolver } from "@hookform/resolvers/zod"
import { Check, ChevronsUpDown } from "lucide-react"
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
      <form onSubmit={form.handleSubmit(onSubmit)} className="flex flex-col space-y-6">
        {/*++++++++++++++++++Origin & Destination++++++++++++++++++*/}
        <div className="flex flex-col sm:flex-row sm:space-x-4 space-y-4 sm:space-y-0 w-full">
          <FormField
            control={form.control}
            name="origin"
            render={({ field }) => (
              <FormItem className="flex flex-col w-full">
                <FormLabel>Origin <span className="text-destructive">*</span></FormLabel>
                <Popover>
                  <PopoverTrigger asChild>
                    <FormControl>
                      <Button
                        variant="outline"
                        role="combobox"
                        className={cn(
                          "w-full justify-between",
                          !field.value && "text-muted-foreground"
                        )}
                      >
                        {field.value
                          ? stations.find(
                            (station) => station.value === field.value
                          )?.label
                          : "Select Origin"}
                        <ChevronsUpDown className="opacity-50" />
                      </Button>
                    </FormControl>
                  </PopoverTrigger>
                  <PopoverContent align="start" sideOffset={4} className="w-[var(--radix-popover-trigger-width)] min-w-0 max-w-full p-0">
                    <Command>
                      <CommandInput
                        placeholder="Search Origin..."
                        className="h-9"
                      />
                      <CommandList>
                        <CommandEmpty>No framework found.</CommandEmpty>
                        <CommandGroup>
                          {stations.map((station) => (
                            <CommandItem
                              value={station.label}
                              key={station.value}
                              onSelect={() => {
                                form.setValue("origin", station.value)
                              }}
                              disabled={form.watch("destination") === station.value}
                            >
                              {station.label}
                              <Check
                                className={cn(
                                  "ml-auto",
                                  station.value === field.value
                                    ? "opacity-100"
                                    : "opacity-0"
                                )}
                              />
                            </CommandItem>
                          ))}
                        </CommandGroup>
                      </CommandList>
                    </Command>
                  </PopoverContent>
                </Popover>
                {/* <FormDescription>
                  This is the language that will be used in the dashboard.
                </FormDescription> */}
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="destination"
            render={({ field }) => (
              <FormItem className="flex flex-col w-full">
                <FormLabel>Destination <span className="text-destructive">*</span></FormLabel>
                <Popover>
                  <PopoverTrigger asChild>
                    <FormControl>
                      <Button
                        variant="outline"
                        role="combobox"
                        className={cn(
                          "w-full justify-between",
                          !field.value && "text-muted-foreground"
                        )}
                      >
                        {field.value
                          ? stations.find(
                            (station) => station.value === field.value
                          )?.label
                          : "Select destination"}
                        <ChevronsUpDown className="opacity-50" />
                      </Button>
                    </FormControl>
                  </PopoverTrigger>
                  <PopoverContent align="start" sideOffset={4} className="w-[var(--radix-popover-trigger-width)] min-w-0 max-w-full p-0">
                    <Command>
                      <CommandInput
                        placeholder="Search Destination..."
                        className="h-9"
                      />
                      <CommandList>
                        <CommandEmpty>No destination found.</CommandEmpty>
                        <CommandGroup>
                          {stations.map((station) => (
                            <CommandItem
                              value={station.label}
                              key={station.value}
                              onSelect={() => {
                                form.setValue("destination", station.value)
                              }}
                              disabled={form.watch("origin") === station.value}
                            >
                              {station.label}
                              <Check
                                className={cn(
                                  "ml-auto",
                                  station.value === field.value
                                    ? "opacity-100"
                                    : "opacity-0"
                                )}
                              />
                            </CommandItem>
                          ))}
                        </CommandGroup>
                      </CommandList>
                    </Command>
                  </PopoverContent>
                </Popover>
                {/* <FormDescription>
                  This is the language that will be used in the dashboard.
                </FormDescription> */}
                <FormMessage />
              </FormItem>
            )}
          />
        </div>
        <div className="">
          <FormField
            control={form.control}
            name="datetime"
            render={({ field }) => (
              <FormItem className="flex w-72 flex-col gap-2">
                <FormLabel htmlFor="datetime">Date time <span className="text-destructive">*</span></FormLabel>
                <FormControl>
                  <DateTimePicker value={field.value} onChange={field.onChange} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>
        <Button type="submit">Search</Button>
      </form>
    </Form>
  )
}
