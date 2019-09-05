loss.backward()

if args.switch:
    # update Ce matrices using ``Bucket Switching`` scheme
    for name, m in net.named_modules():
        if not hasattr(m, 'mask'):
            continue
        with torch.no_grad():
            qC = m.sparsify_and_quantize_C()
            # grad_C = m.C.grad
            dC = optim.get_d(m.C)##?
            if dC is None:
                continue
            if args.dC_threshold > 0.0:

                dC[dC.abs() <= args.dC_threshold] = 0.0
            m.C.grad = None
            dC_sign = dC.sign().float()
            # update ``dC_counter``
            m.dC_counter.add_(dC_sign)
            activated = m.dC_counter.abs() == args.switch_bar
            # if activated.any():
            #     print('Ce is updated!!')
            dC_sign = m.dC_counter.sign() * activated.float()
            # Ce non-zero and gradient non-zero
            dC_pow = dC_sign * qC.sign().float()
            dC_mul = 2 ** dC_pow
            # Ce zero (not in the mask) and gradient non-zero
            dC_add = (qC == 0.0).float() * m.mask * dC_sign * args.min_C
            # update C
            new_C = qC.data * dC_mul + dC_add
            if args.max_C is not None:
                new_C.clamp_(-args.max_C, args.max_C)
            m.C.data = new_C
            # reset activated counters to 0
            m.dC_counter[activated] = 0.0
            # m.C.data = sparsify_and_nearestpow2(new_C, args.threshold)

optim.step()