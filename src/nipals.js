
export function nipals(X, Y, u) {
  let diff = 1;
  let t, q, w, tOld;
  for (var i = 0; diff > 1e-10; i++) {
    w = X.transpose().mmul(u).div(u.transpose().mmul(u).get(0, 0));
    // console.log('w without norm', JSON.stringify(w));
    w = w.div(w.norm());
    // console.log('w', JSON.stringify(w));
    // console.log(w.transpose().mmul(w));
    // calc X scores
    t = X.mmul(w).div(w.transpose().mmul(w).get(0, 0));// t_h paso 3
    // calc loading
    // console.log('scores', t);
    if (i > 0) {
      diff = t.clone().sub(tOld).pow(2).sum();
      // console.log('diff', diff);
    }
    tOld = t.clone();
    // Y block, calc weights, normalise and calc Y scores
    // steps can be omitted for 2 class Y (simply by setting q_h=1)
    q = Y.transpose().mmul(t).div(t.transpose().mmul(t).get(0, 0));
    q = q.div(q.norm());

    u = Y.mmul(q).div(q.transpose().mmul(q).get(0, 0));
  }
  return { t, w, q, u };
}
