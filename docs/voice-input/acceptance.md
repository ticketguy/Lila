# Acceptance checklist

Record the date, OS build, WSL version/distribution, keyboard model, microphone
source and test result. Keep automated and physical hardware results separate.

## Automated

- [ ] .NET 10 desktop build succeeds for Release.
- [ ] .NET shortcut/session/lifecycle tests pass.
- [ ] Python voice capture/ASR adapter tests pass.
- [ ] Python authenticated voice API tests pass.
- [ ] Full Python unit suite passes.
- [ ] Linux activation shell script passes `bash -n` and idempotency/deactivation
      review.
- [ ] `git diff --check` passes.

## Install and service

- [ ] Windows activation publishes to the current user's LocalAppData without
      administrator elevation.
- [ ] Running activation twice is safe; an unmarked destination is preserved.
- [ ] Windows startup remains off unless explicitly enabled in Settings.
- [ ] Linux activation checks prerequisites and starts the user service.
- [ ] Repeated Linux activation is safe; deactivation removes only its own
      unchanged service unit.
- [ ] Credentials, settings, models, recordings and transcripts survive
      deactivation.
- [ ] API is authenticated and loopback-only; ready state is truthful.
- [ ] WSL helper/service failure is reported and recovery can be retried.

## Keyboard and voice hardware

- [ ] Right Ctrl starts after the configured delay and release submits once.
- [ ] Right Ctrl 1/3/5/10 second holds approximate the physical hold duration.
- [ ] Ctrl+C, Ctrl+V, Ctrl+S and ordinary typing remain unaffected.
- [ ] Fn+P/Pause first press starts; key-up does not stop; second press submits.
- [ ] Escape cancels both activation types without transcript submission.
- [ ] Repeats, fast events, missing release and keyboard device removal do not
      create duplicate/stuck sessions.
- [ ] Test while the app is focused, hidden in tray, and another app is focused.
- [ ] Microphone produces valid mono 16 kHz audio; Parakeet returns transcript;
      exact transcript reaches LilaCore.
- [ ] Closing the main window keeps the tray/service lifecycle active; Exit
      cancels voice and shuts down its managed resources.

## Hardware report boundary

The T450s Pause pulse is an observed compatibility fact. This repository change
has not been independently rerun on the T450s in this implementation run. Mark
each physical result only after doing the test on that machine. Other laptops,
Fn mappings, audio bridges and microphones remain unvalidated until tested.
