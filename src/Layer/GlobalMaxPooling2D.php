<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalMaxPooling2D extends AbstractGlobalMaxPooling
{
    protected $rank = 2;

    public function __construct(object $backend,array $options=null)
    {
        $leftargs = [];
        extract($this->extractArgs([
            'name'=>null,
        ],$options,$leftargs));
        $this->initName($name,'globalmaxpooling2d');
        parent::__construct($backend, $leftargs);
    }
}
