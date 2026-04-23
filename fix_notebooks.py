import json

def fix_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        source = "".join(cell.get('source', []))
        
        if 'def discover_multi_topology_upgraded' in source:
            # Replace the string in the source
            old_str = """def discover_multi_topology_upgraded(train_triples, num_relations, min_support=8, min_conf=0.1):
    \"\"\"
    升级版：同时返回置信度，用于作为权重参与规则融合打分
    \"\"\"
    pairs_by_r = defaultdict(set)
    for h, r, t in train_triples:
        pairs_by_r[r].add((h, t))
        
    rules_chain, rules_vstruct, rules_fork = {}, {}, {}
    
    for r_i in range(num_relations):
        pi = pairs_by_r[r_i]
        if not pi: continue
        
        h_to_x = defaultdict(list)
        x_to_h = defaultdict(list)
        for h, x in pi:
            h_to_x[h].append(x)
            x_to_h[x].append(h)
            
        for r_j in range(num_relations):
            pj = pairs_by_r[r_j]
            if not pj: continue
            
            x_to_t = defaultdict(list)
            t_to_x = defaultdict(list)
            for x, t in pj:
                x_to_t[x].append(t)
                t_to_x[t].append(x)
                
            chain_pairs, vstruct_pairs, fork_pairs = set(), set(), set()
            
            for h, xs in h_to_x.items():
                for x in xs:
                    for t in x_to_t[x]:
                        chain_pairs.add((h, t))
                        
            for x, hs in x_to_h.items():
                if x in t_to_x:
                    for h in hs:
                        for t in t_to_x[x]:
                            vstruct_pairs.add((h, t))
                            
            for x, hs in h_to_x.items():
                if x in x_to_t:
                    for h in hs:
                        for t in x_to_t[x]:
                            fork_pairs.add((h, t))
                            
            def eval_rule(pairs, rule_dict):
                sup = len(pairs)
                if sup < min_support: return
                best_rk, best_hit = None, -1
                for r_k in range(num_relations):
                    hit = len(pairs & pairs_by_r[r_k])
                    if hit > best_hit:
                        best_rk, best_hit = r_k, hit
                conf = best_hit / sup
                if conf >= min_conf:
                    rule_dict[(r_i, r_j)] = (best_rk, conf) # 保存最佳目标关系和对应的置信度
                    
            eval_rule(chain_pairs, rules_chain)
            eval_rule(vstruct_pairs, rules_vstruct)
            eval_rule(fork_pairs, rules_fork)
            
    return rules_chain, rules_vstruct, rules_fork"""
            new_str = """def discover_multi_topology_upgraded(train_triples, num_relations, min_support=8, min_conf=0.1):
    \"\"\"
    升级版：同时返回置信度，用于作为权重参与规则融合打分
    \"\"\"
    pairs_by_r = defaultdict(set)
    for h, r, t in train_triples:
        pairs_by_r[r].add((h, t))
        
    rules_chain, rules_fork1, rules_fork2, rules_rev_chain = {}, {}, {}, {}
    
    for r_i in range(num_relations):
        pi = pairs_by_r[r_i]
        if not pi: continue
        
        h_to_x = defaultdict(list)
        x_to_h = defaultdict(list)
        for h, x in pi:
            h_to_x[h].append(x)
            x_to_h[x].append(h)
            
        for r_j in range(num_relations):
            pj = pairs_by_r[r_j]
            if not pj: continue
            
            x_to_t = defaultdict(list)
            t_to_x = defaultdict(list)
            for x, t in pj:
                x_to_t[x].append(t)
                t_to_x[t].append(x)
                
            chain_pairs, fork1_pairs, fork2_pairs, rev_chain_pairs = set(), set(), set(), set()
            
            for x, heads in x_to_h.items():
                if x in x_to_t:
                    tails = x_to_t[x]
                    for h in heads:
                        for t in tails:
                            chain_pairs.add((h, t))
                            
            for x, heads_i in x_to_h.items():
                if x in t_to_x:
                    heads_j = t_to_x[x]
                    for h in heads_i:
                        for t in heads_j:
                            fork1_pairs.add((h, t))
                            
            for x, tails_i in h_to_x.items():
                if x in x_to_t:
                    tails_j = x_to_t[x]
                    for h in tails_i:
                        for t in tails_j:
                            fork2_pairs.add((h, t))
                            
            for x, tails_i in h_to_x.items():
                if x in t_to_x:
                    heads_j = t_to_x[x]
                    for h in tails_i:
                        for t in heads_j:
                            rev_chain_pairs.add((h, t))
                            
            def eval_rule(pairs, rule_dict):
                sup = len(pairs)
                if sup < min_support: return
                best_rk, best_hit = None, -1
                for r_k in range(num_relations):
                    hit = len(pairs & pairs_by_r[r_k])
                    if hit > best_hit:
                        best_rk, best_hit = r_k, hit
                conf = best_hit / sup
                if conf >= min_conf:
                    rule_dict[(r_i, r_j)] = (best_rk, conf) # 保存最佳目标关系和对应的置信度
                    
            eval_rule(chain_pairs, rules_chain)
            eval_rule(fork1_pairs, rules_fork1)
            eval_rule(fork2_pairs, rules_fork2)
            eval_rule(rev_chain_pairs, rules_rev_chain)
            
    return rules_chain, rules_fork1, rules_fork2, rules_rev_chain"""
            source = source.replace(old_str, new_str)
            
            # Now other replacements in the same cell
            source = source.replace("        self.rules_vstruct = defaultdict(list) \n        self.rules_fork = defaultdict(list) ", 
                                    "        self.rules_fork1 = defaultdict(list) \n        self.rules_fork2 = defaultdict(list) \n        self.rules_rev_chain = defaultdict(list) ")
            # And `set_rules` replacement
            source = source.replace("def set_rules(self, sym, inv, chain_r, vstruct_r, fork_r):", "def set_rules(self, sym, inv, chain_r, fork1_r, fork2_r, rev_chain_r):")
            source = source.replace("self.rules_vstruct.clear()\n        self.rules_fork.clear()", "self.rules_fork1.clear()\n        self.rules_fork2.clear()\n        self.rules_rev_chain.clear()")
            source = source.replace("for (r_i, r_j), (r_k, conf) in vstruct_r.items(): self.rules_vstruct[r_k].append((r_i, r_j, conf))\n        for (r_i, r_j), (r_k, conf) in fork_r.items(): self.rules_fork[r_k].append((r_i, r_j, conf))",
                                    "for (r_i, r_j), (r_k, conf) in fork1_r.items(): self.rules_fork1[r_k].append((r_i, r_j, conf))\n        for (r_i, r_j), (r_k, conf) in fork2_r.items(): self.rules_fork2[r_k].append((r_i, r_j, conf))\n        for (r_i, r_j), (r_k, conf) in rev_chain_r.items(): self.rules_rev_chain[r_k].append((r_i, r_j, conf))")
            
            # _compositionality_infer replacement
            comp_old = """
        for r_i, r_j, conf in self.rules_vstruct.get(r_q, []):
            for x in self._tails_by_hr.get((h_q, r_i), []):
                amp_hx = self.W_beta_gamma[h_q, r_i]
                for t in self._heads_by_tr.get((x, r_j), []):
                    amp_tx = self.W_beta_gamma[t, r_j]
                    scores_comp[t] = max(scores_comp.get(t, 0.0), amp_hx * amp_tx * conf * self.alpha_comp)

        for r_i, r_j, conf in self.rules_fork.get(r_q, []):
            for x in self._heads_by_tr.get((h_q, r_i), []):
                amp_xh = self.W_beta_gamma[x, r_i]
                for t in self._tails_by_hr.get((x, r_j), []):
                    amp_xt = self.W_beta_gamma[x, r_j]
                    scores_comp[t] = max(scores_comp.get(t, 0.0), amp_xh * amp_xt * conf * self.alpha_comp)"""
            comp_new = """
        for r_i, r_j, conf in self.rules_fork1.get(r_q, []):
            for x in self._tails_by_hr.get((h_q, r_i), []):
                amp_hx = self.W_beta_gamma[h_q, r_i]
                for t in self._heads_by_tr.get((x, r_j), []):
                    amp_tx = self.W_beta_gamma[t, r_j]
                    scores_comp[t] = max(scores_comp.get(t, 0.0), amp_hx * amp_tx * conf * self.alpha_comp)

        for r_i, r_j, conf in self.rules_fork2.get(r_q, []):
            for x in self._heads_by_tr.get((h_q, r_i), []):
                amp_xh = self.W_beta_gamma[x, r_i]
                for t in self._tails_by_hr.get((x, r_j), []):
                    amp_xt = self.W_beta_gamma[x, r_j]
                    scores_comp[t] = max(scores_comp.get(t, 0.0), amp_xh * amp_xt * conf * self.alpha_comp)
                    
        for r_i, r_j, conf in self.rules_rev_chain.get(r_q, []):
            for x in self._heads_by_tr.get((h_q, r_i), []):
                amp_xh = self.W_beta_gamma[x, r_i]
                for t in self._heads_by_tr.get((x, r_j), []):
                    amp_tx = self.W_beta_gamma[t, r_j]
                    scores_comp[t] = max(scores_comp.get(t, 0.0), amp_xh * amp_tx * conf * self.alpha_comp)"""
            source = source.replace(comp_old, comp_new)
            comp_old2 = """        for r_i, r_j, conf in self.rules_vstruct.get(r_q, []):
            for x in self._tails_by_hr.get((h_q, r_i), []):
                amp_hx = self.W_beta_gamma[h_q, r_i]
                for t in self._heads_by_tr.get((x, r_j), []):
                    amp_tx = self.W_beta_gamma[t, r_j]
                    scores_comp[t] = max(scores_comp.get(t, 0.0), amp_hx * amp_tx * conf * self.alpha_comp)

        for r_i, r_j, conf in self.rules_fork.get(r_q, []):
            for x in self._heads_by_tr.get((h_q, r_i), []):
                amp_xh = self.W_beta_gamma[x, r_i]
                for t in self._tails_by_hr.get((x, r_j), []):
                    amp_xt = self.W_beta_gamma[x, r_j]
                    scores_comp[t] = max(scores_comp.get(t, 0.0), amp_xh * amp_xt * conf * self.alpha_comp)"""
            source = source.replace(comp_old2, comp_new.strip('\n'))
            
            # learn_all_rules replacement
            learn_old = """    def learn_all_rules(self, min_support=8, min_conf=0.1, sym_min_support=2, sym_min_conf=0.1, inv_min_support=2, inv_min_conf=0.1):
        pair_rels, sym, inv = discover_sym_inv_upgraded(self.train_triples, self.M, sym_min_support, sym_min_conf, inv_min_support, inv_min_conf)
        c, v, f = discover_multi_topology_upgraded(self.train_triples, self.M, min_support, min_conf)
        self.set_rules(sym, inv, c, v, f)
        
        all_rules = {}
        for k, (rk, conf) in c.items(): all_rules[f"chain_{k}"] = rk
        for k, (rk, conf) in v.items(): all_rules[f"vstruct_{k}"] = rk
        for k, (rk, conf) in f.items(): all_rules[f"fork_{k}"] = rk
        return all_rules, sym, inv"""
            learn_new = """    def learn_all_rules(self, min_support=8, min_conf=0.1, sym_min_support=2, sym_min_conf=0.1, inv_min_support=2, inv_min_conf=0.1):
        pair_rels, sym, inv = discover_sym_inv_upgraded(self.train_triples, self.M, sym_min_support, sym_min_conf, inv_min_support, inv_min_conf)
        c, f1, f2, rc = discover_multi_topology_upgraded(self.train_triples, self.M, min_support, min_conf)
        self.set_rules(sym, inv, c, f1, f2, rc)
        
        all_rules = {}
        for k, (rk, conf) in c.items(): all_rules[f"chain_{k}"] = rk
        for k, (rk, conf) in f1.items(): all_rules[f"fork1_{k}"] = rk
        for k, (rk, conf) in f2.items(): all_rules[f"fork2_{k}"] = rk
        for k, (rk, conf) in rc.items(): all_rules[f"rev_chain_{k}"] = rk
        return all_rules, sym, inv"""
            source = source.replace(learn_old, learn_new)

            # Update the cell source splitting by newline
            lines = source.splitlines(True)
            cell['source'] = lines

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

fix_notebook('/home/amax/Zixing_Jia/2026_03_model/NSR_Visualization.ipynb')

